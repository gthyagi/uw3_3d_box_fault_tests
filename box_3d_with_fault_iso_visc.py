# ### Case3: Anisotropic with fault

import underworld3 as uw
import numpy as np
import sympy
import os
from enum import Enum
from underworld3 import timing
import pyvista as pv

if uw.mpi.size == 1:
    # to fix trame issue
    import nest_asyncio
    nest_asyncio.apply()
    
    import underworld3.visualisation as vis
    import matplotlib.pyplot as plt
    import cmcrameri.cm as cmc

os.environ["UW_TIMING_ENABLE"] = "1"

# +
# output dir
output_dir = os.path.join(os.path.join("./output/"), 
                          f'box_3d_iso_visc_with_fault')

if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)

# +
# mesh parameter
vdegree  = uw.options.getInt("vdegree", default=2)
pdegree = uw.options.getInt("pdegree", default=1)
pcont = uw.options.getBool("pcont", default=True)
pcont_str = str(pcont).lower()

# solver parameters
stokes_tol = uw.options.getReal("stokes_tol", default=1e-5)
stokes_tol_str = str("{:.1e}".format(stokes_tol))


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

# create mesh
mesh = uw.discretisation.Mesh(f'./output/meshes/box_3d_with_fault.msh', 
                              boundaries=boundaries_3D, 
                              boundary_normals=boundary_normals_3D, 
                              useMultipleTags=True, 
                              useRegions=True, 
                              coordinate_system_type=uw.coordinates.CoordinateSystemType.CARTESIAN)

if uw.mpi.size == 1:
    vis.plot_mesh(mesh)

if uw.mpi.size == 1:
    mesh.view()

# mesh variables
v_soln = uw.discretisation.MeshVariable('V', mesh, mesh.data.shape[1], degree=vdegree)
p_soln = uw.discretisation.MeshVariable('P', mesh, 1, degree=pdegree, continuous=pcont)
strain_rate_inv2 = uw.discretisation.MeshVariable("eps", mesh, 1, degree=1)
fault_dist = uw.discretisation.MeshVariable("df", mesh, 1, degree=2, continuous=False)
fault_norm = uw.discretisation.MeshVariable("nf", mesh, mesh.dim, degree=1, continuous=True, varsymbol=r"{\hat{n}}")

# +
# read fault surface
fault_surf = pv.read("./output/meshes/fault_surface.msh")
if uw.mpi.size == 1:
    fault_surf.plot(style='wireframe', color='k',)
    
fault_surf_poly = fault_surf.extract_surface()
fault_surf_poly.compute_normals(inplace=True)

# +
## Fault distance function (using pyvista)
sample_points = pv.PolyData(fault_dist.coords)
pv_mesh_d = sample_points.compute_implicit_distance(fault_surf_poly)
sample_points.point_data["df"] = pv_mesh_d.point_data["implicit_distance"]

with mesh.access(fault_dist):
    fault_dist.data[:, 0] =sample_points.point_data["df"]

# +
## Map fault normals (computed by pyvista)
fault_kdtree = uw.kdtree.KDTree(fault_surf_poly.points)

with mesh.access(fault_norm):
    closest_points, dist_sq, _ = fault_kdtree.find_closest_point(fault_norm.coords)
    dist = np.sqrt(dist_sq)
    mask = dist < mesh.get_min_radius() * 2.5
    fault_norm.data[mask] = fault_surf_poly.point_data["Normals"][closest_points[mask]]
# -

sample_points

if uw.mpi.size == 1:
    pv_mesh = vis.mesh_to_pv_mesh(mesh)
    pv_mesh.point_data["norm"] = uw.function.evalf(fault_norm.sym, pv_mesh.points)
    pv_mesh.point_data["dist"] = uw.function.evalf(fault_dist.sym, pv_mesh.points)
    
    pv_mesh_clipped = pv_mesh.clip(normal="y", origin=(0,0.5,0))
    
    pl = pv.Plotter(window_size=[1000, 1000])
    
    pl.add_mesh(pv_mesh, style="wireframe")
    pl.add_mesh(pv_mesh_clipped, scalars="dist", cmap="RdBu_r", clim=(-0.5,0.5))
    
    # pl.add_points(sample_points, scalars="df", cmap="RdBu", clim=(-0.5,0.5))
    pl.add_arrows(pv_mesh.points, pv_mesh.point_data["norm"], mag=2)
    pl.add_mesh(fault_surf_poly, style='wireframe', color='k')
    
    pl.show()

# +
## Solver

## Now determine how the problem will be set up: Stokes (viscous) solve to compute stresses
## in which faults will appear as weak zones (could be elastic / damage or viscous / damage)

stokes = uw.systems.Stokes(mesh, velocityField=v_soln, pressureField=p_soln)

stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
stokes.constitutive_model.Parameters.eta_0 = 1
stokes.constitutive_model.Parameters.eta_1 = 0 + sympy.Piecewise(
    (0.001, fault_dist.sym[0] < mesh.get_min_radius() * 2),
    (1, True),
)

stokes.constitutive_model.Parameters.director = fault_norm.sym

stokes.penalty = 1.0
stokes.saddle_preconditioner = sympy.simplify(
    1 / (stokes.constitutive_model.viscosity + stokes.penalty)
)

stokes.add_essential_bc(
    [1, 0, 0],
    mesh.boundaries.Left.name,
)
stokes.add_essential_bc(
    [-1, 0, 0],
    mesh.boundaries.Right.name,
)
stokes.add_essential_bc(
    [None, None, 0],
    mesh.boundaries.Bottom.name,
)
stokes.add_essential_bc(
    [None, 0, None],
    mesh.boundaries.Front.name,
)
stokes.add_essential_bc(
    [None, 0, None],
    mesh.boundaries.Back.name,
)

stokes.bodyforce = -1 * mesh.CoordinateSystem.unit_k

# +
stokes.petsc_options["ksp_monitor"] = None
stokes.petsc_options["snes_monitor"] = None
stokes.petsc_options["snes_atol"] = 1.0e-4

stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "cg"
stokes.petsc_options["fieldsplit_velocity_pc_type"] = "mg"

stokes.petsc_options["fieldsplit_pressure_ksp_type"] = "gmres"
stokes.petsc_options["fieldsplit_pressure_pc_type"] = "mg"

timing.reset()
timing.start()

stokes.solve(zero_init_guess=False)

timing.print_table(display_fraction=0.999)

# +
nodal_strain_rate_inv2 = uw.systems.Projection(mesh, strain_rate_inv2)
nodal_strain_rate_inv2.uw_function = stokes.Unknowns.Einv2
nodal_strain_rate_inv2.smoothing = 1.0e-2
nodal_strain_rate_inv2.petsc_options["ksp_monitor"] = None
nodal_strain_rate_inv2.petsc_options["snes_monitor"] = None

nodal_strain_rate_inv2.solve()
# -

if uw.mpi.size == 1:
    pv_mesh = vis.mesh_to_pv_mesh(mesh)
    pv_mesh.point_data["norm"] = uw.function.evalf(fault_norm.sym, pv_mesh.points)
    pv_mesh.point_data["D"] = uw.function.evalf(fault_dist.sym, pv_mesh.points)
    pv_mesh.point_data["V"] = uw.function.evalf(v_soln.sym, pv_mesh.points)
    pv_mesh.point_data["Edot"] = uw.function.evalf(strain_rate_inv2.sym, pv_mesh.points)
    
    pv_mesh_clip = pv_mesh.clip(normal="y", origin=(0, 0.5, 0))
    
    fault_surf_poly.point_data["Edot"] = uw.function.evalf(strain_rate_inv2.sym, fault_surf_poly.points)
    fault_surf_poly.point_data["V"] = uw.function.evalf(v_soln.sym, fault_surf_poly.points)
    
    
    pl = pv.Plotter(window_size=[1000, 1000])
    
    pl.add_mesh(pv_mesh, style="wireframe", color="Grey", opacity=0.25)
    # pl.add_mesh(pv_mesh_clip, style="surface", scalars="D", cmap="RdBu")
    
    # pl.add_points(sample_points, scalars="df", cmap="RdBu", clim=(-0.5,0.5))
    # pl.add_arrows(pv_mesh.points, pv_mesh.point_data["norm"], mag=0.1)
    
    pl.add_arrows(pv_mesh.points, pv_mesh.point_data["V"], mag=0.2, opacity=0.3)
    
    pl.add_mesh(fault_surf_poly, cmap="RdBu_r", scalars="Edot", clim=(0,0.75))
    #pl.add_arrows(surf2d.points, surf2d.point_data["V"], mag=0.3)
    
    pl.show()

# saving h5 and xdmf file
mesh.petsc_save_checkpoint(index=0, meshVars=[v_soln, p_soln, strain_rate_inv2, fault_dist, fault_norm], outputPath=os.path.relpath(output_dir)+'/output')


