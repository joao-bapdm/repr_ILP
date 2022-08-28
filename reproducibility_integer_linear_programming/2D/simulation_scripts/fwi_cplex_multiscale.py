from firedrake import *

import os
import time
import json
import argparse
import spyro
import spyro.optimizers.damp as adam
from spyro.optimizers.tobs import TOBS

from mpi4py import MPI
import numpy.linalg as la
import numpy as np

# Load parameters
model = spyro.io.load_model()
# model["parallelism"]["type"] = "off"

# Create the computational environment
comm = spyro.utils.mpi_init(model)

# Create mesh
mesh, V = spyro.utils.create_mesh(model, comm, quad=False)
print(f"Function space with {V.dim()} DoFs")

# load data
if model["data"].get("initfile"):
    vp_guess = spyro.utils.load_velocity_model(
        model, V, source_file=model["data"]["initfile"]
    )
else:
    vp_guess = interpolate(Constant(model["opts"]["cmin"]), V)
File("vp_init.pvd").write(vp_guess)
vp_gradient = Function(V)

# load exact reference model
if model["data"].get("exactfile"):
    vp_exact = spyro.utils.load_velocity_model(
        model, V, source_file=model["data"]["exactfile"]
    )
else:
    vp_exact = None

# get control from vp
normalized_vp = spyro.utils.normalize_vp(model, vp_guess)
discrete_data = normalized_vp.dat.data.round()
normalized_vp.dat.data[:] = discrete_data
File("before_vp_normalized.pvd").write(normalized_vp)

# acquisition geometry
sources, receivers = spyro.Geometry(model, mesh, V, comm).create()

# import IPython; IPython.embed(); exit()

# get water dofs
water = spyro.utils.water_layer(
    mesh,
    V,
    vp_guess,
    model
)

# Set water layer to zero
normalized_vp.dat.data[water] = 0
File("after_vp_normalized.pvd").write(normalized_vp)

# excitation frequency
FREQ = model["acquisition"]["frequency"]

# Specify output
resultpic, fobjfile = model["data"]["pic"], model["data"]["fobj"]
outdir, resultfile = model["output"]["outdir"], model["data"]["resultfile"]

i# Initialize output log
os.makedirs(model["output"]["outdir"], exist_ok=True)
with open(os.path.join(model["output"]["outdir"], "output.log"), "w") as f: 
    f.write("OPTIMIZATION HISTORY")
    f.write("\n"+"===================="+"\n")

# Define a callback function that returns the gradient and functional
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
    if freq_band:
        print(
            "INFO: Executing inversion for low-passed cut off of "
            + str(freq_band)
            + " Hz...",
            flush=True,
        )

    # Spatial communicator rank and size.
    rank = comm.comm.rank
    size = comm.comm.size

    # Update control xi from rank 0.
    xi = COMM_WORLD.bcast(xi, root=0)

    # Update the local vp_guess/control function
    # NOTE TO SELF: this should become a func
    # n = len(vp_guess.dat.data[:])
    n = len(normalized_vp.dat.data[:])
    N = [comm.comm.bcast(n, r) for r in range(size)]
    indices = np.insert(np.cumsum(N), 0, 0)
    normalized_vp.dat.data[:] = xi[indices[rank] : indices[rank + 1]]
    vp_guess.assign(spyro.utils.control_to_vp(model, normalized_vp))

    # Check if the program has converged (and exit if so).
    stops[0] = COMM_WORLD.bcast(stop[0], root=0)

    dJ_total = np.zeros((len(xi),), dtype=float)
    dJ_local = np.zeros((len(xi),), dtype=float)
    J_local = np.array([0.0])
    J_total = np.array([0.0])

    if stops[0] == 0:
        for sn in range(model["acquisition"]["num_sources"]):
            # if sn != 1: continue

            # if sn != 0:
            #     print(f"skipping {sn}")
            #     continue

            if spyro.io.is_owner(comm, sn):
                # Load in "exact" or "observed" shot records from a pickle.
                shotfile = model["data"]["shots"]+str(FREQ)+"Hz_sn_"+str(sn)+".dat"
                p_exact_recv = spyro.io.load_shots(shotfile)
                # low-pass filter the shot record for the current frequency band.
                if freq_band:
                    p_exact_recv = spyro.utils.butter_lowpass_filter(
                        p_exact_recv, freq_band, 1.0 / model["timeaxis"]["dt"]
                    )
                # Compute the forward simulation for "guess".
                p_guess, p_guess_recv = spyro.solvers.Leapfrog(
                    # model, mesh, comm, vp_guess, sources, receivers,
                    # source_num=sn
                    model, mesh, comm, spyro.utils.control_to_vp(model,
                        normalized_vp), sources, receivers, source_num=sn, lp_freq_index=index)
                # Calculate the misfit.
                misfit = spyro.utils.evaluate_misfit( model, comm,
                        p_guess_recv, p_exact_recv)
                # Calculate the gradient of the functional.
                dJ = spyro.solvers.Leapfrog_adjoint(
                    # model, mesh, comm, vp_guess, p_guess, misfit,
                    # source_num=sn
                    model, mesh, comm, spyro.utils.control_to_vp(model,
                        normalized_vp), p_guess, misfit, source_num=sn)
                dJ_local += dJ
                # Calculate the L2-functional.
                J = spyro.utils.compute_functional(model, comm, misfit)
                J_local[0] += J

        # if comm.ensemble_comm.rank == 0:
        #     # Plot updated control variable
        #     viz_m_file.write(normalized_vp)
        #     # Plot updated P-wave velocity
        #     viz_vp_file.write(vp_guess)

    # Sum functional and gradient over ensemble members
    comm.ensemble_comm.Allreduce(dJ_local, dJ_total, op=MPI.SUM)
    comm.ensemble_comm.Allreduce(J_local, J_total, op=MPI.SUM)

    # Mask the vertices that are in the water
    # dJ_total[water]=0.0
    dJ_total_io = Function(V, val=dJ_total)

    # Filter gradient
    if model["cplex"]["use_rmin"]: dJ_total_io = spyro.utils.helmholtz_filter(
            dJ_total_io, model['opts']['rmin'])
    dJ_total = dJ_total_io.dat.data[:]
    dJ_total[water]=0.0

    vp_gradient.assign(dJ_total_io)
    # if comm.ensemble_comm.rank == 0:
    #     viz_dm_file.write(vp_gradient)

    # write paraview output
    cb.write_file(m=normalized_vp, dm=vp_gradient, vp=vp_guess)
    if vp_exact: M.append(errornorm(vp_exact, vp_guess))
    spyro.io.save_image(
        vp_guess, fname=os.path.join(
            model["output"]["outdir"], "intermediate_" + model["data"]["pic"]
        )
    )

    return J_total[0], dJ_total

# Load cplex parameters
max_iter = model["inversion"]["max_iter"]
beta = model["cplex"]["beta"]
mul_beta = model["cplex"]["mul_beta"]
mul_rmin = model["cplex"]["mul_rmin"]
lim_rmin = model["cplex"]["lim_rmin"]
# gbar = model["cplex"]["gbar"]
gbar = 1
epsilons = model["cplex"]["epsilons"]
iter_info = "it.: {:d} | obj.f.: {:e} | rel.var.: {:2.2f}% |"
iter_info += " move: {:g} | rmin: {:g}"

# ADAM parameters
gamma_m = model["cplex"].get("gamma_m", 0.5)
gamma_v = model["cplex"].get("gamma_v", 0.5)
m, v = 0, 0

# Initialize quality measure and objective function measures
M, fobj = [], []

for index, freq_band in enumerate(model["inversion"]["freq_bands"]):
    if not freq_band:
        freq_band = ""
    with open(os.path.join(model["output"]["outdir"], "output.log"), "a") as f: 
        if comm.comm.rank == 0 and comm.ensemble_comm.rank == 0:
            f.write("\n"+"--------------------")
            f.write(f"inverting for lowpassed {freq_band}Hz signal")
            f.write("--------------------")
    # Callback object for output files
    name = str(freq_band)+"Hz" if freq_band else ""
    cb = spyro.io.Callback(model, comm, name=name)
    cb.create_file(normalized_vp, vp_gradient, vp_guess)
    # Gather the full velocity data on the master rank
    xi = normalized_vp.vector().gather()
    # Bounds for control
    if model["material"]["type"] == "simp":
        lb = 0
        ub = 1
    elif model["material"]["type"] == None:
        lb = model["opts"]["cmin"]
        ub = model["opts"]["cmax"]

    stop = [0]
    change = 100
    counter = 0


    # Call the optimization routine from the master rank.
    if comm.comm.rank == 0 and comm.ensemble_comm.rank == 0:

        while counter < max_iter:

            # evaluate functional, gradient
            model['opts']['rmin'] = COMM_WORLD.bcast(model['opts']['rmin'], root=0)
            J, dJ = shots(xi, stop)
            fobj.append(J)
            # dJ /= la.norm(dJ)

            # get first moment
            m = adam.moving_avg(gamma_m, m, dJ)
            # second moment
            v = adam.moving_avg(gamma_v, v, dJ ** 2)

            # correct for bias
            m_ = adam.remove_bias(gamma_m, m, counter)
            v_ = adam.remove_bias(gamma_v, v, counter)

            # evaluate RMSProp averaged gradient
            dJ = adam.RMSprop(m_, v_)

            # exit if close enough, or if objective function stagnates
            if J < 1e-16 or (counter > 0 and J == J0): break

            if counter > 0:
                # calcula a variação na função objetivo
                change = (J - J0) / J0

            # update beta
            beta = spyro.optimizers.update_flip_limits( beta, counter, mul_beta,
                    change, xi, mode='counter')
            # update rmin
            model['opts']['rmin'] = spyro.optimizers.update_rmin(
                    model['opts']['rmin'], counter, lim_rmin, mul_rmin)

            # update control
            xi = TOBS(
                dJ,
                # np.ones(dJ.shape) / mesh.num_cells(),
                np.zeros(dJ.shape),
                gbar,
                xi.mean(),
                epsilons,
                beta,
                xi)

            # printa diferença entre iterações
            print("\n"+iter_info.format( counter, J, change, beta,
                model['opts']['rmin'])+"\n")
            # print to file
            with open(os.path.join(model["output"]["outdir"], "output.log"), "a") as f: 
                f.write("\n"+iter_info.format( counter, J, change, beta,
                        model['opts']['rmin'])+"\n")

            # save old functional
            J0, dJ0 = J, dJ

            counter += 1

        stop = [1]
        model['opts']['rmin'] = COMM_WORLD.bcast(model['opts']['rmin'],root=0)
        shots(xi, stop)

        # print to stdout
        print("\n"+iter_info.format( counter, J, change, beta,
                model['opts']['rmin'])+"\n")
        # print to file
        with open(os.path.join(model["output"]["outdir"], "output.log"), "a") as f: 
            f.write("\n"+iter_info.format( counter, J, change, beta,
                    model['opts']['rmin'])+"\n")

    else:
        while stop[0] == 0:
            model['opts']['rmin'] = COMM_WORLD.bcast(model['opts']['rmin'], root=0)
            shots(xi, stop)

    # Retrieve values
    spyro.utils.spatial_scatter(comm, xi, normalized_vp)
    # Update control xi from rank 0
    xi = COMM_WORLD.bcast(xi, root=0)
    normalized_vp.dat.data[:] = xi[:]
    vp_guess = spyro.utils.control_to_vp(model, normalized_vp)

    if freq_band:
        model["data"]["pic"] = str(freq_band) + "Hz_" + resultpic
        model["data"]["resultfile"] = str(freq_band) + "Hz_" + resultfile
        model["data"]["fobj"] = str(freq_band) + "Hz_"  + fobjfile

    # Save hdf5 and png of final result
    spyro.utils.save_velocity_model(comm, vp_guess, os.path.join(outdir, model["data"]["resultfile"]))
    spyro.io.save_image(vp_guess, fname=os.path.join(outdir, model["data"]["pic"]))
    # Register objective function history
    if comm.comm.rank == 0 and comm.ensemble_comm.rank == 0:
        np.save(os.path.join(outdir, model["data"]["fobj"]), np.array(fobj))
        if vp_exact:
            if freq_band:
                quali = str(freq_band)+"Hz_"+ "quality_measure"
            else:
                quali = "quality_measure"
            np.save(os.path.join(outdir, quali), np.array(M))

# config file used
if not isinstance(model["acquisition"]["source_pos"], list):
    model["acquisition"]["source_pos"] = model["acquisition"]["source_pos"].tolist()
if not isinstance(model["acquisition"]["receiver_locations"], list):
    model["acquisition"]["receiver_locations"] = model["acquisition"]["receiver_locations"].tolist()

spyro.io.save_model(
    model, 
    jsonfile=os.path.join(
        model["output"]["outdir"], 
        model["data"]["configfile"]
    )
)
