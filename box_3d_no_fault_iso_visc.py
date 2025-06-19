# ### Case1: Non refined mesh with no fault

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
                          f'box_3d_no_fault_iso_visc')

if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)

# +
# mesh parameter
res = uw.options.getReal("res", default=1/4) # 8, 16, 32, 64, 128
cellsize = 1/res
minX, minY, minZ = 0., 0., -40.
maxX, maxY, maxZ = 150., 150., 0.


vdegree  = uw.options.getInt("vdegree", default=2)
pdegree = uw.options.getInt("pdegree", default=1)
pcont = uw.options.getBool("pcont", default=True)
pcont_str = str(pcont).lower()

# solver parameters
stokes_tol = uw.options.getReal("stokes_tol", default=1e-5)
stokes_tol_str = str("{:.1e}".format(stokes_tol))
# -

# create mesh
mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(minX, minY, minZ), maxCoords=(maxX, maxY, maxZ),
                                         cellSize=cellsize, filename=f'{output_dir}/mesh.msh')

if uw.mpi.size == 1:
    vis.plot_mesh(mesh)

if uw.mpi.size == 1:
    mesh.view()

# mesh variables
v_soln = uw.discretisation.MeshVariable('V', mesh, mesh.data.shape[1], degree=vdegree)
p_soln = uw.discretisation.MeshVariable('P', mesh, 1, degree=pdegree, continuous=pcont)
strain_rate_inv2 = uw.discretisation.MeshVariable("eps", mesh, 1, degree=1)

# Create Stokes object
stokes = uw.systems.Stokes(mesh, velocityField=v_soln, pressureField=p_soln)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0
stokes.saddle_preconditioner = 1.0

# +
# set up boundary conditions
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

# +
# saving h5 and xdmf file

mesh.petsc_save_checkpoint(index=0, meshVars=[v_soln, p_soln, strain_rate_inv2], 
                           outputPath=os.path.relpath(output_dir)+'/output')

# # does not work for this model
# mesh.write_timestep('output', index=0, outputPath=os.path.relpath(output_dir),
#                     meshVars=[v_soln, p_soln, strain_rate_inv2], meshUpdates=True,)

# +
mesh_h5_name = f'{os.path.relpath(output_dir)}/output_mesh.h5'

# Write mesh
mesh_h5 = mesh.write(mesh_h5_name, index=0)
uw.mpi.barrier()
# -
# Write variable fields (each should be collective)
v_soln.write(f'{os.path.relpath(output_dir)}/v_sol.mesh.V.00000.h5')
v_soln.save(mesh_h5_name, name='V') # Save variables to main mesh file
uw.mpi.barrier()


# Write variable fields (each should be collective)
p_soln.write(f'{os.path.relpath(output_dir)}/p_sol.mesh.P.00000.h5')
p_soln.save(mesh_h5_name, name='P') # Save variables to main mesh file
uw.mpi.barrier()

# Write variable fields (each should be collective)
strain_rate_inv2.write(f'{os.path.relpath(output_dir)}/sr_inv.mesh.eps.00000.h5')
strain_rate_inv2.save(mesh_h5_name, name='eps') # Save variables to main mesh file
uw.mpi.barrier()

# Generate XDMF after all data are flushed
mesh.generate_xdmf(mesh_h5_name)
uw.mpi.barrier()


