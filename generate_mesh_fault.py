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
    gmsh.model.add("Box")
    
    # Create domain
    sx, sy, sz = startCoords
    ex, ey, ez = endCoords
    v_domain = gmsh.model.occ.add_box(sx, sy, sz, ex - sx, ey - sy, ez - sz)
    
    # creating fault line
    f_pt_s_x, f_pt_s_y, f_pt_s_z = fault_s_point
    f_pt_e_x, f_pt_e_y, f_pt_e_z = fault_e_point
    f_pt_s = gmsh.model.occ.add_point(f_pt_s_x, f_pt_s_y, f_pt_s_z)
    f_pt_e = gmsh.model.occ.add_point(f_pt_e_x, f_pt_e_y, f_pt_e_z)
    f_line = gmsh.model.occ.add_line(f_pt_s, f_pt_e)
    
    # extrude fault line to create fault surface
    dimTags = gmsh.model.occ.extrude([(1, f_line)], 
                                     fault_depth * math.cos(math.radians(fault_angle)), 
                                     fault_depth * 0.0, 
                                     fault_depth * math.sin(math.radians(fault_angle)), 
                                     # numElements=[100] # controls no. of in fault 
                                    )
    fault_surf_tag = dimTags[1][1]
    # Embed faults into domain so cells will align along fault
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
    
    # (Optional) print for verification
    print("Mapped faces:", surface_map)
    
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


if __name__ == "__main__":
    Box3DwithFault(startCoords=(startX, startY, startZ), 
                   endCoords=(endX, endY, endZ),
                   filename=f'{output_dir}/mesh.msh',
                   boundaries=boundaries_3D,
                   DX_FAULT=1.5,
                   DX_BIAS=1.12,
                   gui=False)


