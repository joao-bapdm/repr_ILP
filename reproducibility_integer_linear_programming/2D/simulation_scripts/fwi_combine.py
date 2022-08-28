from datetime import datetime
import os

from firedrake import Function, File, COMM_WORLD, errornorm, Constant
from firedrake.petsc import PETSc
import numpy as np

from spyro.optimizers.optimisation import optimisation
import spyro

###############################################################
#################### PROPAGATOR DEFINITION ####################
###############################################################
counter = 0
def shots(xi, stops):
    """A callback function that returns gradient of control
    and functional to be optimized using scipy

    Parameters
    ----------
    xi: array-like
        The control vector to be optimized
    stops: list of integer
        0 to terminate optimization, 1 to continue

    Returns
    -------
    J: float
        Functional
    dJ: array-like
        The gradient of the functional w.r.t. to the velocity model

    """
    
    # Spatial communicator rank and size.
    rank = comm.comm.rank
    size = comm.comm.size
    # Ensemble communicator rank and size.
    ens_rank = comm.ensemble_comm.rank
    ens_size = comm.ensemble_comm.size

    # Update control xi from rank 0.
    xi_salt, xi_background = np.split(xi, 2)
    xi_salt = COMM_WORLD.bcast(xi_salt, root=0)
    xi_background = COMM_WORLD.bcast(xi_background, root=0)

    # Update the local vp_guess/control function
    penal=Constant(model["material"]["penal"])
    salt_penal=Constant(model["material"]["salt_penal"])
    spyro.utils.spatial_scatter(comm, xi_background, control_background)
    spyro.utils.spatial_scatter(comm, xi_salt, control_salt)
    spyro.utils.material_model(vp_guess, {"salt": control_salt, "background": control_background}, model, p=penal, salt_p=salt_penal)
    File("vp_guess"+str(counter)+".pvd").write(vp_guess)

    # Check if the program has converged (and exit if so).
    stops[0] = COMM_WORLD.bcast(stops[0], root=0)

    # Initialize functional variables
    dJ_total_salt = np.zeros((len(gradient_background.dat.data[:]),), dtype=float)
    dJ_total_background = np.zeros((len(gradient_background.dat.data[:]),), dtype=float)
    dJ_local_salt = np.zeros((len(gradient_background.dat.data[:]),), dtype=float)
    dJ_local_background = np.zeros((len(gradient_background.dat.data[:]),), dtype=float)
    J_local = np.array([0.0])
    J_total = np.array([0.0])

    # Loop over shots
    g_rank = comm.global_comm.rank
    if stops[0] == 0:
        for sr in range(len(shot_record)):
            shot_id = ens_rank + sr * ens_size
    
            # Forward problem
            p_guess, guess_record = spyro.solvers.Leapfrog(model, mesh, comm, vp_guess, sources, receivers, source_num=shot_id, lp_freq_index=index)
            # import IPython; IPython.embed()
            # Calculate misfit
            misfit = spyro.utils.evaluate_misfit(model, comm, guess_record, shot_record[sr])
            # Calculate gradient (by the adjoint method)
            dJ = spyro.solvers.Leapfrog_adjoint(
                model, mesh, comm, vp_guess, p_guess, misfit, source_num=shot_id, control_salt=control_salt, control_background=control_background
            )
            dJ_background, dJ_salt = dJ
            ######################################
            dJ_local_salt += dJ_salt.dat.data[:]
            dJ_local_background += dJ_background.dat.data[:]
            if model["salt"].get("exclusive"):
                print("considering only salt information")
                dJ_local_background[:] = 0
            ######################################
            # Calculate the functional
            J = spyro.utils.compute_functional(model, comm, misfit)
            J_local[0] += J

    # Sum functional and gradient over ensemble members
    comm.ensemble_comm.Allreduce(dJ_local_salt, dJ_total_salt)
    comm.ensemble_comm.Allreduce(dJ_local_background, dJ_total_background)
    comm.ensemble_comm.Allreduce(J_local, J_total)

    # import IPython; IPython.embed(); exit()

    # Get gradient Function back
    if dJ_total_salt.any(): gradient_salt.dat.data[:] = dJ_total_salt
    if dJ_total_background.any(): gradient_background.dat.data[:] = dJ_total_background
    # Apply filter to gradient
    spyro.utils.helmholtz_filter(gradient_salt, model["cplex"]["rmin"])
    spyro.utils.helmholtz_filter(gradient_background, model["cplex"]["rmin"])
    # Neutralize water region
    gradient_salt.dat.data[water] = 0.0
    gradient_background.dat.data[water] = 0.0
    # Exclude salt region from background gradient
    gradient_background.dat.data[np.where(control_salt.dat.data == 1)] = 0

    # Measure quality of solution
    if vp_exact: quality_measure.append(errornorm(vp_exact, vp_guess))
    # Register current objective function
    objective_function.append(J_total)

    # Output
    Cb_background.write_file(m=control_background, dm=gradient_background, vp=vp_guess)
    Cb_salt.write_file(m=control_salt, dm=gradient_salt)
    spyro.utils.save_velocity_model(comm, vp_guess, result_hdf5)

    gradient = np.concatenate((gradient_salt.vector().gather(), gradient_background.vector().gather()))
    # Gather (a single) gradient on global rank 0
    #if dJ_total_salt.any() and dJ_total_background.any():
    #    gradient = np.concatenate((gradient_salt.vector().gather(), gradient_background.vector().gather()))
    #elif dJ_total_salt.any() and not dJ_total_background.any():
    #    gradient = gradient_salt.vector().gather()
    #elif not dJ_total_salt.any() and dJ_total_background.any():
    #    gradient = gradient_background.vector().gather()
    #else:
    #    print("No gradient to be used!")
    #    gradient = np.array([])

    return J_total, gradient

###############################################################
############### END OF PROPAGATOR DEFINITION ##################
###############################################################

########## LOAD PARAMETERS ##########
# Time tracker
begin_time = datetime.now() 
# configuration file
model = spyro.io.load_model()
# velocity models
exact_model = model["data"]["exactfile"]
initial_model = model["data"]["initfile"]
salt_model = model["data"]["saltfile"]
background_model = model["data"]["backgroundfile"]
# output files
model["output"]["outdir"] = os.path.join(model["output"]["outdir"], begin_time.strftime("%d%b%y_%-Hh%-Mm%-Ss"))
directory = model["output"]["outdir"]
logfile = os.path.join(directory, "output.log")
fobj_npy = os.path.join(directory, model["data"]["fobj"])
result_png = os.path.join(directory, model["data"]["pic"])
result_hdf5 = os.path.join(directory, model["data"]["resultfile"])

########## INITIALIZE OBJECTS AND VARIABLES ##########
# MPI communicator and mesh
comm = spyro.utils.mpi_init(model)
mesh, V = spyro.utils.create_mesh(model, comm, quad=False)
PETSc.Sys.Print("Function space with %g DoFs" % (V.dim()))
# Velocity models
vp_guess = spyro.utils.load_velocity_model(model, V, file=initial_model)
vp_exact = spyro.utils.load_velocity_model(model, V, file=exact_model)
# Acquisition geometry
sources, receivers = spyro.Geometry(model, mesh, V, comm).create()
# Control and gradient
control_salt = spyro.utils.load_velocity_model(model, V, file=salt_model)
control_salt.dat.data[:] = model["cplex"]["amin"]
control_background = spyro.utils.load_velocity_model(model, V, file=background_model)
gradient_salt = Function(V)
gradient_background = Function(V)

########## WATER DOFS ##########
# Get water (just for gradient)
water, no_water = spyro.utils.water_layer(mesh, V, vp_guess, model)
# no_water, water = spyro.utils.water_layer(mesh, V, vp_guess, model)

########## LOGGING ##########
# Create directory for output
os.makedirs(directory, exist_ok=True)
print(f"Creating directory {directory} for storage of results")
# Start logging output
with open(logfile, "w") as f:
    f.write("OPTIMIZATION HISTORY")
    f.write("\n"+"===================="+"\n")
# Register lists history of objective function and quality measure
quality_measure, objective_function = [], []

########## LOAD SHOT RECORD ##########
# Set shot record to be loaded
shot_file = model["data"]["shots"] + "_" + str(model["acquisition"]["frequency"]) + "Hz"
# Load shot record
shot_record = spyro.io.load_shot(shot_file, len(sources), comm.ensemble_comm)
base_record = shot_record.copy()

########## GATHER CONTROL TO ROOT PROCESS ##########
xi_salt = control_salt.vector().gather()
xi_background = control_background.vector().gather()
xi = np.concatenate((xi_salt, xi_background))
# xi = xi_salt
# Get bounds
lb, ub = 0, 1

try:
    ########## MULTISCALE OPTIMIZATION ##########
    # frequency loop
    for index, freq_band in enumerate(model["inversion"]["freq_bands"]):
        # Register current frequency
        spyro.io.log_new_frequency(freq_band, logfile)
    
        # Filter shot record (if there is a cutting frequency)
        if freq_band:
            for i in range(len(shot_record)):
                shot_record[i] = spyro.utils.butter_lowpass_filter(
                    base_record[i], 
                    freq_band, 
                    1.0 / model["timeaxis"]["dt"]
                )
    
        # Set name for ouput
        suffix_for_pvd = "_" + str(freq_band) + "Hz" if freq_band else ""
        # Create Callback object for pvd writing
        Cb_salt = spyro.io.Callback(model, comm, name=suffix_for_pvd + "_salt")
        Cb_background = spyro.io.Callback(model, comm, name=suffix_for_pvd + "_background")
        Cb_salt.create_file(control_salt, gradient_salt)
        Cb_background.create_file(control_background, gradient_background, vp_guess)
    
        optimisation(model, xi, (lb, ub), shots, comm, logfile)
        # opts = [0]
        # shots(xi, opts)
except OSError: 
    print("something went wrong with the optimization")
finally:
    ########## SAVE OPTIMIZATION INFO ##########
    total_time = datetime.now() - begin_time
    np.save(fobj_npy, np.array(objective_function))
    np.save(os.path.join(directory, "quality"), np.array(quality_measure))
    spyro.io.save_model(model, jsonfile=directory + str("/config.json"))
    spyro.io.save_image(vp_guess, fname=result_png)

    if comm.comm.rank == 0 and comm.ensemble_comm.rank == 0:
        with open(logfile, "a") as f:
            f.write(f"\ntotal execution time = {total_time.seconds} seconds\n")

"""
########## MULTISCALE OPTIMIZATION ##########
# frequency loop
for index, freq_band in enumerate(model["inversion"]["freq_bands"]):
    # Register current frequency
    spyro.io.log_new_frequency(freq_band, logfile)

    # Filter shot record (if there is a cutting frequency)
    if freq_band:
        for i in range(len(shot_record)):
            shot_record[i] = spyro.utils.butter_lowpass_filter(
                base_record[i], 
                freq_band, 
                1.0 / model["timeaxis"]["dt"]
            )

    # Set name for ouput
    suffix_for_pvd = "_" + str(freq_band) + "Hz" if freq_band else ""
    # Create Callback object for pvd writing
    Cb_salt = spyro.io.Callback(model, comm, name=suffix_for_pvd + "_salt")
    Cb_background = spyro.io.Callback(model, comm, name=suffix_for_pvd + "_background")
    Cb_salt.create_file(control_salt, gradient_salt)
    Cb_background.create_file(control_background, gradient_background, vp_guess)

    optimisation(model, xi, (lb, ub), shots, comm, logfile)
    # opts = [0]
    # shots(xi, opts)
"""


