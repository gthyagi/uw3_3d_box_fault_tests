# ### Generate gmsh file
#
# This workflow is primarily based on PyLith.

import gmsh
import numpy as np
from enum import Enum
import os
from typing import Optional, Tuple
import sympy
import math
from dataclasses import dataclass

# output dir
output_dir = os.path.join(os.path.join("./output/"), f'meshes')
os.makedirs(output_dir, exist_ok=True)

# mesh parameter
startX, startY, startZ = 0., 0., 0.
endX, endY, endZ = 150., 150., -40.


# +
class boundaries_3D(Enum):
    Bottom = 11
    Top = 12
    Right = 13
    Left = 14
    Front = 15
    Back = 16
    Fault = 17

class boundary_normals_3D(Enum):
    Bottom = sympy.Matrix([0, 0, 1])
    Top = sympy.Matrix([0, 0, 1])
    Right = sympy.Matrix([1, 0, 0])
    Left = sympy.Matrix([1, 0, 0])
    Front = sympy.Matrix([0, 1, 0])
    Back = sympy.Matrix([0, 1, 0])


# -

@dataclass
class BoundaryGroup:
    name: str
    tag: int
    dim: int
    entities: list

    def create_physical_group(self, recursive=False):
        create_group(self.name, self.tag, self.dim, self.entities, recursive)


@dataclass
class MaterialGroup:
    tag: int
    entities: list

    def create_physical_group(self):
        create_material(self.tag, self.entities)


def create_material(tag, entities):
    if tag <= 0:
        raise ValueError(f"ERROR: Attempting to use non-positive material tag '{tag}'. Tags for physical groups must be positive.")
    dim = gmsh.model.get_dimension()
    name = gmsh.model.get_physical_name(dim, tag)
    if name:
        raise ValueError(f"ERROR: Attempting to use material tag '{tag}' that is already in use for material '{name}'.")
    gmsh.model.addPhysicalGroup(dim, entities, tag)
    gmsh.model.setPhysicalName(dim, tag, f"material-id:{tag}")


def create_group(name, tag, dim, entities, recursive=True, exclude=None):
    gmsh.model.add_physical_group(dim, entities, tag)
    gmsh.model.set_physical_name(dim, tag, name)
    entities_lowerdim = []
    for entity in entities:
        entities_up, entities_down = gmsh.model.get_adjacencies(dim, entity)
        entities_lowerdim += [e for e in entities_down]
    if recursive and dim >= 1:
        create_group(name, tag, dim-1, entities_lowerdim)


def get_math_progression(field_distance, min_dx, bias):
    """
    Generate the Gmsh MathEval string corresponding to the cell size as a function
    of distance, starting cell size, and bias factor.

    The expression is min_dx * bias**n, where n is the number of cells from the fault.
    n = log(1+distance/min_dx*(bias-1))/log(bias)

    In finding the expression for `n`, we make use that the sum of a geometric series with n
    terms Sn = min_dx * (1 + bias + bias**2 + ... + bias**n) = min_dx * (bias**n - 1)/(bias - 1).
    """
    return f"{min_dx}*{bias}^(Log(1.0+F{field_distance}/{min_dx}*({bias}-1.0))/Log({bias}))"


def Box3DwithFault(
    startCoords: Tuple = (0., 0., 0.),
    endCoords: Tuple = (150., 150., -40.),
    fault_s_point = (75., 25., 0.0),
    fault_e_point = (75., 125., 0.0),
    fault_angle=-45,
    fault_depth=30,
    DX_FAULT = 1.5,
    DX_BIAS = 1.02,
    gmsh_verbosity=0,
    boundaries=None,
    filename=None,
    gui=False
):
    """
    Generates a 3-dimensional box mesh with fault.

    DX_FAULT: float = 1.5
    Target element size directly on the fault surface.
    
    DX_BIAS: float = 1.02
    Geometric progression factor controlling the grading of elements away from the fault.
    """    
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
    gmsh.model.add("Box3D")
    
    # Create domain
    sx, sy, sz = startCoords
    ex, ey, ez = endCoords
    v_domain = gmsh.model.occ.add_box(sx, sy, sz, ex - sx, ey - sy, ez - sz)
    
    # creating fault line
    f_pt_s = gmsh.model.occ.add_point(*fault_s_point)
    f_pt_e = gmsh.model.occ.add_point(*fault_e_point)
    f_line = gmsh.model.occ.add_line(f_pt_s, f_pt_e)
    
    # extrude fault line to create fault surface
    dimTags = gmsh.model.occ.extrude([(1, f_line)], 
                                     fault_depth * math.cos(math.radians(fault_angle)), 
                                     fault_depth * 0.0, 
                                     fault_depth * math.sin(math.radians(fault_angle)), 
                                     # numElements=[100] # controls no. of in fault 
                                    )
    fault_surf_tag = dimTags[1][1]
    # # Embed faults into domain so cells will align along fault
    gmsh.model.occ.fragment([(3, v_domain)], [(2, fault_surf_tag)], removeTool=True)
    gmsh.model.occ.synchronize()

    # Create a material for the domain.
    # The tag argument specifies the integer tag for the physical group.
    # The entities argument specifies the array of surfaces for the material.
    materials = (MaterialGroup(tag=1, entities=[v_domain]),)
    for material in materials:
        material.create_physical_group()

    # Get all 2D faces on the volume’s boundary
    _, surfaces = gmsh.model.get_adjacencies(3, v_domain)
    
    # Fetch the box extents
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(3, v_domain)
    
    # Prepare mapping
    tol = 1e-5
    surface_map: dict[boundaries, int] = {}
    
    # Centroid-based detection
    for surf in surfaces:
        sxmin, symin, szmin, sxmax, symax, szmax = gmsh.model.getBoundingBox(2, surf)
        # centroid coords
        cx = 0.5 * (sxmin + sxmax)
        cy = 0.5 * (symin + symax)
        cz = 0.5 * (szmin + szmax)
    
        # check nearest box-plane
        if abs(cx - xmin) < tol:
            surface_map[boundaries.Left] = surf
            continue
        if abs(cx - xmax) < tol:
            surface_map[boundaries.Right] = surf
            continue
        if abs(cy - ymin) < tol:
            surface_map[boundaries.Front] = surf
            continue
        if abs(cy - ymax) < tol:
            surface_map[boundaries.Back] = surf
            continue
        if abs(cz - zmin) < tol:
            surface_map[boundaries.Bottom] = surf
            continue
        if abs(cz - zmax) < tol:
            surface_map[boundaries.Top] = surf
            continue
        # else: it’s not one of the six box faces
    
    # Include the fault surface explicitly
    surface_map[boundaries.Fault] = fault_surf_tag
    
    # # (Optional) print for verification
    # print("Mapped faces:", surface_map)
    
    # Create one BoundaryGroup per detected face
    for side, tag in surface_map.items():
        BoundaryGroup(
            name=side.name,
            tag=side.value,
            dim=2,
            entities=[tag]
        ).create_physical_group()

    gmsh.model.occ.synchronize()

    # We turn off the default sizing methods.
    gmsh.option.set_number("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.set_number("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.set_number("Mesh.MeshSizeExtendFromBoundary", 0)

    # First, we setup a field `field_distance` with the distance from the fault.
    field_distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(field_distance, "SurfacesList", [fault_surf_tag])

    # Second, we setup a field `field_size`, which is the mathematical expression
    # for the cell size as a function of the cell size on the fault, the distance from
    # the fault (as given by `field_size`, and the bias factor.
    # The `GenerateMesh` class includes a special function `get_math_progression` 
    # for creating the string with the mathematical function.
    field_size = gmsh.model.mesh.field.add("MathEval")
    math_exp = get_math_progression(field_distance, min_dx=DX_FAULT, bias=DX_BIAS)
    gmsh.model.mesh.field.setString(field_size, "F", math_exp)

    # Finally, we use the field `field_size` for the cell size of the mesh.
    gmsh.model.mesh.field.setAsBackgroundMesh(field_size)

    # Generate Mesh
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Laplace2D")
    if filename:
        gmsh.write(filename)
    if gui:
        gmsh.fltk.run()

    gmsh.finalize()


# +
# if __name__ == "__main__":
#     Box3DwithFault(startCoords=(startX, startY, startZ), 
#                    endCoords=(endX, endY, endZ),
#                    filename=f'{output_dir}/box_3d_with_fault.msh',
#                    boundaries=boundaries_3D,
#                    DX_FAULT=1.5,
#                    DX_BIAS=1.12,
#                    gmsh_verbosity=1,
#                    gui=False)
# -

def Box3DwithFaultVolume(
    startCoords: Tuple = (0., 0., 0.),
    endCoords: Tuple = (150., 150., -40.),
    fault_s_point = (75., 25., 0.0),
    fault_e_point = (75., 125., 0.0),
    fault_angle=-45,
    fault_depth=30,
    DX_FAULT = 1.5,
    DX_BIAS = 1.02,
    gmsh_verbosity=0,
    boundaries=None,
    filename=None,
    gui=False
):
    """
    Generates a 3-dimensional box mesh with fault.

    DX_FAULT: float = 1.5
    Target element size directly on the fault surface.
    
    DX_BIAS: float = 1.02
    Geometric progression factor controlling the grading of elements away from the fault.
    """    
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
    gmsh.model.add("Box3D")
    
    # Create domain
    sx, sy, sz = startCoords
    ex, ey, ez = endCoords
    v_domain = gmsh.model.occ.add_box(sx, sy, sz, ex - sx, ey - sy, ez - sz)
    
    # creating fault line
    f_pt_s = gmsh.model.occ.add_point(*fault_s_point)
    f_pt_e = gmsh.model.occ.add_point(*fault_e_point)
    f_line = gmsh.model.occ.add_line(f_pt_s, f_pt_e)
    
    # extrude fault line to create fault surface
    dimTags = gmsh.model.occ.extrude([(1, f_line)], 
                                     fault_depth * math.cos(math.radians(fault_angle)), 
                                     fault_depth * 0.0, 
                                     fault_depth * math.sin(math.radians(fault_angle)), 
                                     # numElements=[100] # controls no. of in fault 
                                    )
    fault_surf_tag = dimTags[1][1]

    # Extrude surface to create a fault volume (one element thick)
    fault_volume_extrude = gmsh.model.occ.extrude(
        [(2, fault_surf_tag)],
        1., 0., 0.,
        numElements=[1],     # one element thick
        recombine=False
    )
    fault_volume = fault_volume_extrude[1][1]
    
    # Fragment with domain
    gmsh.model.occ.fragment([(3, v_domain)], [(3, fault_volume)], removeTool=True, removeObject=True)
    gmsh.model.occ.synchronize()


    # ----- CREATE PHYSICAL GROUPS FOR BOTH VOLUMES -----
    # Get all volume tags after fragment
    volumes = [v[1] for v in gmsh.model.getEntities(dim=3)]
    
    if len(volumes) != 2:
        print(f"Warning: Expected 2 volumes, found {len(volumes)}")
    
    # Get volumes and their sizes
    vol_sizes = [(v, gmsh.model.occ.getMass(3, v)) for v in volumes]
    
    # Sort by size: largest first (host), smallest second (fault)
    vol_sizes_sorted = sorted(vol_sizes, key=lambda x: x[1], reverse=True)
    
    host_tag = vol_sizes_sorted[0][0]
    fault_tag = vol_sizes_sorted[1][0]
    
    # Assign physical groups
    host_group = gmsh.model.addPhysicalGroup(3, [host_tag])
    gmsh.model.setPhysicalName(3, host_group, "HostVolume")
    
    fault_group = gmsh.model.addPhysicalGroup(3, [fault_tag])
    gmsh.model.setPhysicalName(3, fault_group, "FaultVolume")


    # Fetch box extents from host volume
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(3, host_tag)
    tol = 1e-5

    # Function to map boundary surfaces for a given volume
    def get_surface_map(volume_tag, boundaries, fault_surf_tag=None):
        _, surfaces = gmsh.model.get_adjacencies(3, volume_tag)
        surface_map = {}
    
        for surf in surfaces:
            sxmin, symin, szmin, sxmax, symax, szmax = gmsh.model.getBoundingBox(2, surf)
            cx = 0.5 * (sxmin + sxmax)
            cy = 0.5 * (symin + symax)
            cz = 0.5 * (szmin + szmax)
    
            if abs(cx - xmin) < tol:
                surface_map[boundaries.Left] = surf
                continue
            if abs(cx - xmax) < tol:
                surface_map[boundaries.Right] = surf
                continue
            if abs(cy - ymin) < tol:
                surface_map[boundaries.Front] = surf
                continue
            if abs(cy - ymax) < tol:
                surface_map[boundaries.Back] = surf
                continue
            if abs(cz - zmin) < tol:
                surface_map[boundaries.Bottom] = surf
                continue
            if abs(cz - zmax) < tol:
                surface_map[boundaries.Top] = surf
                continue
    
        if fault_surf_tag is not None:
            surface_map[boundaries.Fault] = fault_surf_tag
    
        return surface_map

    # Get surface maps for host and fault
    host_surface_map = get_surface_map(host_tag, boundaries)
    fault_surface_map = get_surface_map(fault_tag, boundaries, fault_surf_tag=fault_surf_tag)

    # --- COLLECT all upper faces for combined group ---
    host_upper = host_surface_map.get(boundaries.Top)
    fault_upper = None

    # For the fault volume, the "top" face may not always be identified by the above logic.
    # So, find by centroid z ~ zmax (coplanar with host top)
    _, fault_surfaces = gmsh.model.get_adjacencies(3, fault_tag)
    for surf in fault_surfaces:
        sxmin, symin, szmin, sxmax, symax, szmax = gmsh.model.getBoundingBox(2, surf)
        cz = 0.5 * (szmin + szmax)
        if abs(cz - zmax) < tol:
            fault_upper = surf
            break

    # # Now create a single physical group for the UpperSurface if both found
    # upper_faces = []
    # if host_upper: upper_faces.append(host_upper)
    # if fault_upper: upper_faces.append(fault_upper)
    # if upper_faces:
    #     group_tag = gmsh.model.addPhysicalGroup(2, upper_faces)
    #     gmsh.model.setPhysicalName(2, group_tag, "UpperSurface")
    
    upper_faces = []
    if host_upper: upper_faces.append(host_upper)
    if fault_upper: upper_faces.append(fault_upper)
    if upper_faces:
        group_tag = gmsh.model.addPhysicalGroup(2, upper_faces, boundaries.Top.value)
        gmsh.model.setPhysicalName(2, group_tag, boundaries.Top.name)


    # Create boundary groups for all faces in host (except top, which is now combined)
    for side, tag in host_surface_map.items():
        if side == boundaries.Top:
            continue
        BoundaryGroup(
            name=side.name,
            tag=side.value,
            dim=2,
            entities=[tag]
        ).create_physical_group()

    # Do the same for the fault volume (except upper, which is combined, and maybe except Fault if you already group it elsewhere)
    for side, tag in fault_surface_map.items():
        if (fault_upper is not None and tag == fault_upper) or side == boundaries.Top:
            continue
        BoundaryGroup(
            name=side.name,
            tag=side.value,
            dim=2,
            entities=[tag]
        ).create_physical_group()

    gmsh.model.occ.synchronize()

    # We turn off the default sizing methods.
    gmsh.option.set_number("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.set_number("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.set_number("Mesh.MeshSizeExtendFromBoundary", 0)

    # First, we setup a field `field_distance` with the distance from the fault.
    field_distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(field_distance, "SurfacesList", [fault_surf_tag])

    # Second, we setup a field `field_size`, which is the mathematical expression
    # for the cell size as a function of the cell size on the fault, the distance from
    # the fault (as given by `field_size`, and the bias factor.
    # The `GenerateMesh` class includes a special function `get_math_progression` 
    # for creating the string with the mathematical function.
    field_size = gmsh.model.mesh.field.add("MathEval")
    math_exp = get_math_progression(field_distance, min_dx=DX_FAULT, bias=DX_BIAS)
    gmsh.model.mesh.field.setString(field_size, "F", math_exp)

    # Finally, we use the field `field_size` for the cell size of the mesh.
    gmsh.model.mesh.field.setAsBackgroundMesh(field_size)

    # Generate Mesh
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Laplace2D")

    if filename:
        gmsh.write(filename)
    if gui:
        gmsh.fltk.run()

    gmsh.finalize()

if __name__ == "__main__":
    Box3DwithFaultVolume(startCoords=(startX, startY, startZ), 
                   endCoords=(endX, endY, endZ),
                   filename=f'{output_dir}/box_3d_with_faultvolume.msh',
                   boundaries=boundaries_3D,
                   DX_FAULT=1.5,
                   DX_BIAS=1.12,
                   gmsh_verbosity=1,
                   gui=True)

0/0


def extract_fault_surface(input_msh: str,
                          output_msh: str,
                          fault_phys_name: str = "Fault"):
    gmsh.initialize()
    gmsh.open(input_msh)

    # 1) find the 2D Physical Group named fault_phys_name
    phys_groups = gmsh.model.getPhysicalGroups(dim=2)
    fault_group = None
    for (dim, tag) in phys_groups:
        name = gmsh.model.getPhysicalName(dim, tag)
        if name == fault_phys_name:
            fault_group = tag
            break
    if fault_group is None:
        gmsh.fltk.run()
        raise ValueError(f"No 2D Physical Group named '{fault_phys_name}'")

    # 2) find all entities that belong to that group
    fault_entities = gmsh.model.getEntitiesForPhysicalGroup(2, fault_group)

    for (dim, tag) in gmsh.model.getEntities():
        if dim == 2 and tag in fault_entities:
            continue
        if dim >= 2:
            gmsh.model.removeEntities([(dim, tag)], recursive=True)

    # 4) write and finalize
    gmsh.write(output_msh)
    gmsh.finalize()


def remove_fault_group(
    input_msh: str,
    output_msh: str,
    fault_phys_name: str = "Fault"
):
    """
    Loads a Gmsh .msh, removes all mesh entities assigned to the
    2D Physical Group named `fault_phys_name`, and writes the result.
    """
    gmsh.initialize()
    gmsh.open(input_msh)

    # 1) Find the 2D Physical Group tag with the given name
    fault_tag = None
    for (dim, tag) in gmsh.model.getPhysicalGroups(dim=2):
        name = gmsh.model.getPhysicalName(2, tag)
        if name == fault_phys_name:
            fault_tag = tag
            break

    if fault_tag is None:
        gmsh.finalize()
        raise ValueError(f"No 2D Physical Group named '{fault_phys_name}' found")

    # 2) Get all 2D entities in that Physical Group
    fault_entities = gmsh.model.getEntitiesForPhysicalGroup(2, fault_tag)

    # 3) Remove those surface entities (but leave 3D volume and other surfaces intact)
    for tag in fault_entities:
        gmsh.model.removeEntities([(2, tag)], recursive=False)

    # 4) Remove the Physical Group definition itself
    gmsh.model.removePhysicalGroups([(2, fault_tag)])

    # 5) Write out the modified mesh
    gmsh.write(output_msh)
    gmsh.finalize()


def RefinedBox3DaroundFault(
    startCoords: Tuple[float, float, float]   = (0.0,   0.0,   0.0),
    endCoords:   Tuple[float, float, float]   = (150.0, 150.0, -40.0),
    fault_s_point: Tuple[float, float, float] = (75.0,  25.0,   0.0),
    fault_e_point: Tuple[float, float, float] = (75.0, 125.0,   0.0),
    fault_angle:  float                        = -45.0,
    fault_depth:  float                        = 30.0,
    DX_FAULT:     float                        = 1.5,
    DX_BIAS:      float                        = 1.02,
    gmsh_verbosity: int                        = 0,
    filename:     Optional[str]                = None,
    gui:          bool                         = False,
    boundaries = None
):
    """
    Generates a 3D box mesh with a planar fault surface, refines the mesh around
    the fault without fragmenting the volume, tags material and boundaries, and
    writes the mesh to file or opens the GUI.

    Parameters
    ----------
    DX_FAULT : float
        Target element size on the fault.
    DX_BIAS : float
        Grading bias away from the fault.
    """
    # 1) Initialize
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
    gmsh.model.add("BoxWithFault")

    # 2) Create box volume
    sx, sy, sz = startCoords
    ex, ey, ez = endCoords
    v_domain = gmsh.model.occ.add_box(sx, sy, sz,
                                      ex - sx, ey - sy, ez - sz)

    # 3) Create fault line and extrude to surface
    ps = gmsh.model.occ.add_point(*fault_s_point)
    pe = gmsh.model.occ.add_point(*fault_e_point)
    fl = gmsh.model.occ.add_line(ps, pe)
    dx = fault_depth * math.cos(math.radians(fault_angle))
    dz = fault_depth * math.sin(math.radians(fault_angle))
    dimTags = gmsh.model.occ.extrude([(1, fl)], dx, 0.0, dz)
    fault_surf_tag = dimTags[1][1]

    # 4) Synchronize CAD kernel
    gmsh.model.occ.synchronize()

    # 5) Tag the volume with a MaterialGroup
    mat = MaterialGroup(tag=1, entities=[v_domain])
    mat.create_physical_group()

    # 6) Identify and tag boundary faces by centroid location
    _, surfs = gmsh.model.get_adjacencies(3, v_domain)
    xmin, ymin, zmin, xmax, ymax, zmax = \
        gmsh.model.getBoundingBox(3, v_domain)

    tol = 1e-6
    surface_map: dict[boundaries, int] = {}

    for s in surfs:
        sx0, sy0, sz0, sx1, sy1, sz1 = gmsh.model.getBoundingBox(2, s)
        cx = 0.5*(sx0 + sx1)
        cy = 0.5*(sy0 + sy1)
        cz = 0.5*(sz0 + sz1)

        if abs(cx - xmin) < tol:
            surface_map[boundaries.Left]   = s; continue
        if abs(cx - xmax) < tol:
            surface_map[boundaries.Right]  = s; continue
        if abs(cy - ymin) < tol:
            surface_map[boundaries.Back]   = s; continue
        if abs(cy - ymax) < tol:
            surface_map[boundaries.Front]  = s; continue
        if abs(cz - zmin) < tol:
            surface_map[boundaries.Bottom] = s; continue
        if abs(cz - zmax) < tol:
            surface_map[boundaries.Top]    = s; continue

    # 7) Add the fault surface for boundary tagging
    surface_map[boundaries.Fault] = fault_surf_tag

    # 8) Create BoundaryGroup for each face
    for side, tag in surface_map.items():
        BoundaryGroup(
            name=side.name,
            tag=side.value,
            dim=2,
            entities=[tag]
        ).create_physical_group()

    gmsh.model.occ.synchronize()

    # 9) Disable default sizing
    gmsh.option.set_number("Mesh.MeshSizeFromPoints",    0)
    gmsh.option.set_number("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.set_number("Mesh.MeshSizeExtendFromBoundary", 0)

    # 10) Distance field from fault
    fd = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(fd, "SurfacesList", [fault_surf_tag])

    # 11) Threshold field for refined region around fault
    R_min, R_max = 0.0, fault_depth
    fth = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(fth, "InField", fd)
    gmsh.model.mesh.field.setNumber(fth, "SizeMin", DX_FAULT)
    gmsh.model.mesh.field.setNumber(fth, "SizeMax", DX_FAULT * (DX_BIAS**5))
    gmsh.model.mesh.field.setNumber(fth, "DistMin", R_min)
    gmsh.model.mesh.field.setNumber(fth, "DistMax", R_max)

    # 12) Apply as background mesh
    gmsh.model.mesh.field.setAsBackgroundMesh(fth)

    # 13) Generate, optimize, export, and finalize
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Laplace2D")

    if filename:
        gmsh.write(filename)
    if gui:
        gmsh.fltk.run()
    gmsh.finalize()

    
    # saving fault mesh only
    extract_fault_surface(input_msh=filename, output_msh=f'{output_dir}/fault_surface.msh', 
                          fault_phys_name='Fault')

    # saving refined volume mesh without fault
    remove_fault_group(input_msh=filename, output_msh=f'{output_dir}/refined_box_3d_around_fault.msh',
                       fault_phys_name="Fault")

if __name__ == "__main__":
    RefinedBox3DaroundFault(
        startCoords=(0.0, 0.0, 0.0),
        endCoords=(150.0, 150.0, -40.0),
        fault_s_point=(75.0, 25.0, 0.0),
        fault_e_point=(75.0, 125.0, 0.0),
        fault_angle=-45,
        fault_depth=30,
        DX_FAULT=1.5,
        DX_BIAS=1.3,
        gmsh_verbosity=1,
        boundaries=boundaries_3D,
        filename=f'{output_dir}/refined_box_around_fault_orig.msh',
        gui=False
    )





