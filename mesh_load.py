import underworld3 as uw
import numpy as np
import sympy
import os
from enum import Enum
from underworld3 import timing
import pyvista as pv
import math
import h5py

if uw.mpi.size == 1:
    # to fix trame issue
    import nest_asyncio
    nest_asyncio.apply()
    
    import underworld3.visualisation as vis
    import matplotlib.pyplot as plt
    import cmcrameri.cm as cmc

os.environ["UW_TIMING_ENABLE"] = "1"

# mesh parameter
vdegree  = uw.options.getInt("vdegree", default=2)
pdegree = uw.options.getInt("pdegree", default=1)
pcont = uw.options.getBool("pcont", default=True)
pcont_str = str(pcont).lower()

# input dir
input_dir_ref = os.path.join(os.path.join("./output/"), f'box_3d_no_fault_iso_visc')
input_dir_mesh_non_ref = os.path.join(os.path.join("./output/"), f'meshes')
input_dir_non_ref = os.path.join(os.path.join("./output/"), f'box_3d_with_faultvol_iso_visc')

# +
# # output dir
# output_dir = os.path.join(os.path.join("./output/"), 
#                           f'box_3d_iso_visc_with_faultvolume')

# if uw.mpi.rank == 0:
#     os.makedirs(output_dir, exist_ok=True)
# -

# load mesh
mesh_ref = uw.discretisation.Mesh(f'{input_dir_ref}/mesh.msh.h5')

# +
# create mesh variable
v_soln_ref = uw.discretisation.MeshVariable('V', mesh_ref, mesh_ref.data.shape[1], degree=vdegree)

# load mesh variable
v_soln_ref.read_timestep(data_filename='v_sol', data_name='V', index=0, outputPath=input_dir_ref)
# -

v_soln_ref.view()

# load mesh
mesh_non_ref = uw.discretisation.Mesh(f'{input_dir_mesh_non_ref}/box_3d_with_faultvolume.msh.h5')

# +
# create mesh variable
v_soln_non_ref = uw.discretisation.MeshVariable('V', mesh_non_ref, mesh_non_ref.data.shape[1], degree=vdegree)

# load mesh variable
v_soln_non_ref.read_timestep(data_filename='v_sol', data_name='V', index=0, outputPath=input_dir_non_ref)

# +
# mesh_ref.petsc_save_checkpoint(index=0, meshVars=[v_soln_ref], outputPath=os.path.relpath(input_dir_ref)+'/output_test')
# mesh_non_ref.petsc_save_checkpoint(index=0, meshVars=[v_soln_non_ref], outputPath=os.path.relpath(input_dir_non_ref)+'/output_test')
# -

v_soln_ref2nonref = uw.discretisation.MeshVariable('V_ref', mesh_non_ref, mesh_non_ref.data.shape[1], degree=vdegree)

uw.adaptivity.mesh2mesh_meshVariable(v_soln_ref, v_soln_ref2nonref, )

v_sol_diff_non_ref = uw.discretisation.MeshVariable('V_diff', mesh_non_ref, mesh_non_ref.data.shape[1], degree=vdegree)

with mesh_non_ref.access(v_sol_diff_non_ref):
    v_sol_diff_non_ref.data[...] = uw.function.evalf(v_soln_ref2nonref.sym-v_soln_non_ref.sym, v_sol_diff_non_ref.coord)





# +


# filename = f"{input_dir_ref}/v_sol.mesh.V.00000.h5"
# filename = f"{input_dir_ref}/p_sol.mesh.P.00000.h5"
filename = f"{input_dir_ref}/sr_inv.mesh.eps.00000.h5"

with h5py.File(filename, "r") as f:
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"Group: {name}")
    f.visititems(print_structure)

# +
filename = f"{input_dir_ref}/output_step_00000.h5"

with h5py.File(filename, "r") as f:
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
            if name=='fields/V':
                print(np.array(obj))
        elif isinstance(obj, h5py.Group):
            print(f"Group: {name}")
    f.visititems(print_structure)
# -


