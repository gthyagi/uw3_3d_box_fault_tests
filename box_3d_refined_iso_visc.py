# ### Case1: isoviscous with no fault

import underworld3 as uw
import numpy as np
import sympy
import os
from enum import Enum

if uw.mpi.size == 1:
    # to fix trame issue
    import nest_asyncio
    nest_asyncio.apply()
    
    import pyvista as pv
    import underworld3.visualisation as vis
    import matplotlib.pyplot as plt
    import cmcrameri.cm as cmc

# +
# output dir
output_dir = os.path.join(os.path.join("./output/"), 
                          f'refined_box_3d_iso_visc')

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

class boundary_normals_3D(Enum):
    Bottom = sympy.Matrix([0, 0, 1])
    Top = sympy.Matrix([0, 0, 1])
    Right = sympy.Matrix([1, 0, 0])
    Left = sympy.Matrix([1, 0, 0])
    Front = sympy.Matrix([0, 1, 0])
    Back = sympy.Matrix([0, 1, 0])


# -

# create mesh
mesh = uw.discretisation.Mesh(f'./output/meshes/refined_box_3d_around_fault.msh', 
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
v = uw.discretisation.MeshVariable('V', mesh, mesh.data.shape[1], degree=vdegree)
p = uw.discretisation.MeshVariable('P', mesh, 1, degree=pdegree, continuous=pcont)

# Create Stokes object
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0
stokes.saddle_preconditioner = 1.0

# set up boundary conditions
vel_x = 1.0
stokes.add_dirichlet_bc((vel_x, sympy.oo, sympy.oo), "Left")
stokes.add_dirichlet_bc((-vel_x, sympy.oo, sympy.oo), "Right")

# +
# Stokes settings
stokes.tolerance = stokes_tol
stokes.petsc_options["ksp_monitor"] = None

stokes.petsc_options["snes_type"] = "newtonls"
stokes.petsc_options["ksp_type"] = "fgmres"

# stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")

stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"

stokes.petsc_options[f"fieldsplit_velocity_ksp_type"] = "fcg"
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_max_it"] = 5
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

# # gasm is super-fast ... but mg seems to be bulletproof
# # gamg is toughest wrt viscosity
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "additive")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# mg, multiplicative - very robust ... similar to gamg, additive
stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")
# -

stokes.solve(verbose=True, debug=False)

# saving h5 and xdmf file
mesh.petsc_save_checkpoint(index=0, meshVars=[v, p], outputPath=os.path.relpath(output_dir)+'/output')


