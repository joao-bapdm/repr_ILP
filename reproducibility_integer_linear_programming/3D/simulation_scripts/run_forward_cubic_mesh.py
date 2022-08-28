"""Generate shot records. Use firedrake built-in mesh"""

import firedrake
from firedrake.petsc import PETSc

import spyro


model = spyro.io.load_model()
comm = spyro.utils.mpi_init(model)
# meshing
mesh = firedrake.CubeMesh(
    nx=model["mesh"]["nx"],
    ny=model["mesh"]["ny"],
    nz=model["mesh"]["nz"],
    L=1,
    distribution_parameters={
        "overlap_type": (firedrake.DistributedMeshOverlapType.NONE, 0)
    },
    comm=comm.comm
) 
# function space
V = firedrake.FunctionSpace(
    mesh,
    model["opts"]["method"],
    model["opts"]["degree"]
)
PETSc.Sys.Print(
    f"\nThe mesh has {mesh.num_cells()} elements",
    f" and {mesh.num_vertices()} vertices,\n",
    f"The function space has {V.dim()} DoFs",
    sep="",
    comm=firedrake.COMM_WORLD
    )
# reference velocity model
vp = spyro.io.interpolate(model, mesh, V, fname=model["files"]["input_file"])
firedrake.File("vp_exact.pvd").write(vp)
# acquisition geometry
sources = spyro.Sources(model, mesh, V, comm)
receivers = spyro.Receivers(model, mesh, V, comm)
wavelet = spyro.full_ricker_wavelet(
    dt=model["timeaxis"]["dt"],
    tf=model["timeaxis"]["tf"],
    freq=model["acquisition"]["frequency"],
)
# forward run
p, p_r = spyro.solvers.forward(model, mesh, comm, vp, sources, wavelet, receivers)
for snum in range(model["acquisition"]["num_sources"]):
    if spyro.io.is_owner(comm, snum):
        base_name = model["shots"] + "_" + str(snum) + ".dat"
        print(f"Saving shot {snum} at {base_name} under ensemble rank {comm.ensemble_comm.rank}")
        spyro.io.save_shots(model, comm, p_r, file_name=base_name)
        # spyro.plots.plot_shots(model, comm, p_r, vmin=-5e-2, vmax=5e-2)

