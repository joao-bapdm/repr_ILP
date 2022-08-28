from scipy import optimize
import numpy as np

from spyro.optimizers.tobs import TOBS, TONBS
from spyro.optimizers.cplex import update_flip_limits, update_rmin
import spyro.optimizers.damp as adam

def simplest(model, xi, lb, ub, f, comm):
    """scipy optimization"""

    if comm.comm.rank == 0 and comm.ensemble_comm.rank == 0:
        stop = [0]
        res = optimize.minimize(
            f,
            xi,
            args=(stop,),
            method="L-BFGS-B",
            jac=True,
            bounds=[(lb, ub) for i in range(len(xi))],
            options={"disp": True, "maxiter":model["inversion"]["max_iter"], "gtol": 1e-10}
        )
        stop = [1]
        f(res.x, stop)
        xi = res.x
    else:
        stop = [0]
        while stop[0] == 0:
            f(xi, stop)

    return xi

def simp_plus_tobs(model, xi, lb, ub, f, comm, logfile):
    """SIMP + TOBS"""

    return xi

def tobs(model, xi, lb, ub, f, comm, logfile, ctype=None):
    """TOBS method"""

    # General CPLEX parameters
    rmin = model["cplex"]["rmin"] if model["cplex"]["rmin"] else 0
    beta = model["cplex"]["beta"]
    gbar = model["cplex"]["gbar"]
    mul_beta = model["cplex"]["mul_beta"]
    mul_rmin = model["cplex"]["mul_rmin"]
    lim_rmin = model["cplex"]["lim_rmin"]
    epsilons = model["cplex"]["epsilons"]
    max_iter = model["inversion"]["max_iter"]
    # ADAM parameters
    gamma_m = model["cplex"].get("gamma_m", 0.5)
    gamma_v = model["cplex"].get("gamma_v", 0.5)
    m, v = 0, 0
    # General looping parameters
    stop, change, counter = [0], 100, 0

    # Printing info
    iter_info = "it.: {:d} | obj.f.: {:e} | rel.var.: {:2.2f}% | move: {:g} | rmin: {:g}"


    # Begin optimization loop
    if comm.comm.rank == 0 and comm.ensemble_comm.rank == 0:

        while counter < max_iter:

            J, dJ = f(xi, stop)

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

            # update beta and rmin here??
            beta = update_flip_limits(beta, counter, mul_beta, change, xi, mode='counter')
            rmin = model["cplex"]["rmin"] = update_rmin(rmin, counter, lim_rmin, mul_rmin)


            # update control
            xmin = np.zeros(xi.shape)
            xmax = np.zeros(xi.shape)
            xmin[:xi.size // 2] = model["cplex"].get("amin", 0)
            xmax[:xi.size // 2] = model["cplex"].get("amax", 1)

            optimizer = TONBS if "integer-mixed" in model["material"].get("type") else TOBS
            xi = TOBS(
                dJ,
                np.zeros(dJ.shape),
                gbar,
                xi.mean(),
                epsilons,
                beta,
                xi,
                xmin=xmin,
                xmax=xmax,
                ctype=ctype)

            # print to file
            with open(logfile, "a") as logger:
                logger.write("\n"+iter_info.format(counter, float(J), float(change), beta, rmin)+"\n")

            # save old functional
            J0, dJ0 = J, dJ
            counter += 1

        stop = [1]
        f(xi, stop)

    else:
        while stop[0] == 0:
            model["cplex"]["rmin"] = update_rmin(rmin, counter, lim_rmin, mul_rmin)
            f(xi, stop)

    return xi


def optimisation(model, xi, bounds, f, comm, logfile):
    """Optimization routine for the fwi problem
    
    Parameters
    ----------
    xi: numpy.ndarray
        control vector
    bounds: touple of floats
        specify lower and uper boundary for control values
    f: function
        function corresponding to the forward problem
    comm: communicator
        MPI communicator

    Returns
    -------
    xi: numpy.ndarray
        optimized control vector
    """
    lb, ub = bounds
    method = [model["inversion"]["optimizer"]]

    if isinstance(xi, dict):
        xi = simp_plus_tobs(model, xi, lb, ub, f, comm, logfile)

    elif "scipy_lbfgs" in method:
        xi = simplest(model, xi, lb, ub, f, comm)

    elif "tobs" in method:
        xi = tobs(model, xi, lb, ub, f, comm, logfile)
        # print("entering tobs!")

    elif "mixed-tobs" in method:
        ctype = xi.size // 2 * 'I' + xi.size // 2 * 'C'
        xi = tobs(model, xi, lb, ub, f, comm, logfile, ctype=ctype)

    return xi
