from firedrake import *
from firedrake.petsc import PETSc

import h5py
import copy
from mpi4py import MPI
import numpy as np
import numpy.linalg as la
import math
import scipy.sparse as sp
from scipy.signal import butter, filtfilt
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial.distance import cdist

from ..domains import quadrature

def helmholtz_filter(u, r_min):
    """Smooth scalar field"""

    if r_min:
        V = u.function_space()
        qr_x, _, _ = quadrature.quadrature_rules(V)
    
        # s = Function(V)
        u_ = TrialFunction(V)
        v = TestFunction(V)
        a = r_min**2*inner(grad(u_), grad(v))*dx + u_*v*dx
        L = u*v*dx
        parameters = {'kse_type': 'preonly', 'pctype': 'lu'}
        solve(a == L, u)

    # return u

def build_filter(rmin, midpoint):
    """Build linear filter"""
    
    indptr = [0]
    indices = []
    data = []

    for point in midpoint:
        # get values and location
        wi = np.maximum(0, rmin - la.norm(point - midpoint, axis=1))
        cols = wi.nonzero()[0]

        # store info
        indptr.append(indptr[-1] + cols.size)
        indices += list(cols)
        data += list(wi[wi != 0] / wi.sum())

    return sp.csr_matrix((data, indices, indptr), shape=(len(midpoint), len(midpoint)))

def linear_filter(u, weights):
    """Smooth scalar field"""

    # Get coordinate space
    s = Function(u.function_space(), val = weights @ u.dat.data)

    return s

def butter_lowpass_filter(shot, cutoff, fs, order=2):
    """Low-pass filter the shot record with sampling-rate fs Hz
    and cutoff freq. Hz
    """
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    nr, nc = np.shape(shot)
    filtered_shot = np.zeros((nr, nc))
    for rec, ts in enumerate(shot.T):
        filtered_shot[:, rec] = filtfilt(b, a, ts)
    return filtered_shot


def pml_error(model, p_pml, p_ref):
    """ Erro with PML for a each shot (source) ..."""

    num_sources = model["acquisition"]["num_sources"]
    num_receivers = model["acquisition"]["num_receivers"]
    dt = model["timeaxis"]["dt"]
    tf = model["timeaxis"]["tf"]

    nt = int(tf / dt)  # number of timesteps
    error = []

    for sn in range(num_sources):
        error.append([])
        for ti in range(nt):
            soma = 0
            for rn in range(num_receivers):
                soma += (p_pml[sn][rn][ti] - p_ref[sn][rn][ti]) * (
                    p_pml[sn][rn][ti] - p_ref[sn][rn][ti]
                )
            error[sn].append(math.sqrt(soma / num_receivers))

    return error


def compute_functional(model, comm, residual):
    """ Compute the functional to be optimized """
    num_receivers = model["acquisition"]["num_receivers"]
    dt = model["timeaxis"]["dt"]
    tf = model["timeaxis"]["tf"]
    nt = int(tf / dt)  # number of timesteps

    if comm.comm.rank == 0 and comm.ensemble_comm.rank == 0:
        print("Computing the functional...", flush=True)

    Jtemp = 0.0
    J = 0.0
    Jlist = []
    for ti in range(nt):
        for rn in range(num_receivers):
            Jtemp += 0.5 * (residual[ti][rn] ** 2)
        Jlist.append(Jtemp)
    # Integrate in time (trapezoidal rule)
    for i in range(1, nt - 1):
        J += 0.5 * (Jlist[i - 1] + Jlist[i]) * float(dt)
    J = 0.5 * float(J)
    return J


def evaluate_misfit(model, my_ensemble, guess, exact):
    """Compute the difference between the guess and exact
    at the receiver locations"""

    if "skip" == model["timeaxis"]:
        skip = model["timeaxis"]["skip"]
    else:
        skip = 1

    if my_ensemble.comm.rank == 0 and my_ensemble.ensemble_comm.rank == 0:
        print("Computing the misfit...", flush=True)

    return exact[::skip] - guess


def myrank(COMM=COMM_SELF):
    return COMM.Get_rank()


def mysize(COMM=COMM_SELF):
    return COMM.Get_size()


def mpi_init(model):
    """ Initialize computing environment """
    rank = myrank()
    size = mysize()
    available_cores = COMM_WORLD.size

    if model["parallelism"]["type"] == "automatic":
        num_cores_per_shot = available_cores/model["acquisition"]["num_sources"]
        if available_cores % model["acquisition"]["num_sources"] != 0:
            raise ValueError("Available cores cannot be divided between sources equally.")

    elif model["parallelism"]["type"] == "off":
        num_cores_per_shot = available_cores
    elif model["parallelism"]["type"] == "custom":
        # raise ValueError("Custom parallelism not yet implemented")
        num_cores_per_shot = model["parallelism"]["num_cores_per_shot"]


    comm_ens = Ensemble(COMM_WORLD, num_cores_per_shot)
    return comm_ens


def communicate(array, my_ensemble):
    """Communicate shot record to all processors

    Parameters
    ----------
    array: array-like
        Array of data to all-reduce across both ensemble
        and spatial communicators.
    comm: Firedrake.comm
        A Firedrake ensemble communicator

    Returns
    -------
    array_reduced: array-like
        Array of data max all-reduced
        amongst the ensemble communicator

    """
    array_reduced = copy.copy(array)
    if my_ensemble.comm.size > 1:
        if my_ensemble.comm.rank == 0 and my_ensemble.ensemble_comm.rank == 0:
            print("Spatial parallelism, reducing to comm 0", flush=True)
        my_ensemble.comm.Allreduce(array, array_reduced, op=MPI.MAX)
    return array_reduced


def analytical_solution_for_pressure_based_on_MMS(model, mesh, time):
    degree = model["opts"]["degree"]
    V = FunctionSpace(mesh, "CG", degree)
    z, x = SpatialCoordinate(mesh)
    p = Function(V).interpolate((time ** 2) * sin(pi * z) * sin(pi * x))
    return p


def normalize_vp(model, vp):

    control = firedrake.Function(vp)
    
    if model["opts"].get("control"):
        pass

    elif "material" in model:
        if model["material"]["type"] == "simp":
            vp_min = model["material"]["vp_min"]
            vp_max = model["material"]["vp_max"]
            penal = model["material"]["penal"]
            control.dat.data[:] -= vp_min
            control.dat.data[:] /= (vp_max - vp_min)
            control.dat.data[:] = control.dat.data[:] ** (1 / penal)
            # trim loose ends
            lb, ub = 0, 1
            control.dat.data[:] = np.minimum(
                np.maximum(lb, control.dat.data[:]), ub
            )

    return control


def control_to_vp(model, control):

    vp = firedrake.Function(control)

    if "material" in model:
        if model["material"]["type"] == "simp":
            vp_min = Constant(model["material"]["vp_min"])
            vp_max = Constant(model["material"]["vp_max"])
            penal = Constant(model["material"]["penal"])

            vp.assign(vp_min + (vp_max - vp_min) * control ** penal)

    return vp


def discretize_field(c, bins=None, n=4):

    c_ = firedrake.Function(c)
    vp = c.dat.data
    # Get histogram
    if bins is None:
        counts, bins = np.histogram(c.dat.data, bins=n)
    else:
        counts, _ = np.histogram(c.dat.data, bins=n)

    for i, count in enumerate(counts):
        c_.dat.data[(bins[i] <= vp)&(vp <= bins[i+1])] = (bins[i]+bins[i+1])/2

    return c_, bins


def save_velocity_model(comm, vp_model, dest_file):
    """Save Firedrake.Function representative of a seismic velocity model.
    Stores both nodal values and coordinates into a HDF5 file.

    Parameters
    comm: Firedrake.ensemble_communicator
        The MPI communicator for parallelism
    vp_model: Firedrake.Function
        P-wave seismic velocity interpolated onto the nodes of the
        finite elements.
    dest_file: str
        path to hdf5 file to be written.

    """
    # Sanitize units
    _check_units(vp_model)
    # # Get coordinates
    V = vp_model.function_space()
    W = firedrake.VectorFunctionSpace(V.ufl_domain(), V.ufl_element())
    coords = firedrake.interpolate(V.ufl_domain().coordinates, W)

    # Gather vectors on the master rank
    vp_global = vp_model.vector().gather()
    coords_global = coords.vector().gather()

    if comm.comm.rank == 0 and comm.ensemble_comm.rank == 0:
        print("Writing velocity model: " + dest_file, flush=True)
        with h5py.File(dest_file, "w") as f:
            f.create_dataset("velocity_model", data=vp_global, dtype="f")
            f.create_dataset("coordinates", data=coords_global, dtype="f")
            f.attrs["geometric dimension"] = coords.dat.data.shape[1]
            f.attrs["units"] = "km/s"


def load_velocity_model(params, V, file=None):
    """Load Firedrake.Function representative of a seismic velocity model
    from a HDF5 file.

    Prameters
    ---------
    V: Firedrake.FunctionSpace object
        The space of the finite elements.
    dsource_file: str
        path to hdf5 file to be loaded.

    Returns
    -------
    vp_model: Firedrake.Function
        P-wave seismic velocity interpolated onto the nodes of the
        finite elements.

    """

    if not file:
        if "input" in params:
            if "model" in params["input"]:
                file = params['input']['model']

    if file:
        # Get interpolant
        with h5py.File(file, "r") as f:
            vp = np.asarray(f.get("velocity_model")[()])
            gd = f.attrs['geometric dimension']
            coords = np.asarray(f.get("coordinates")[()])
            coords = coords.reshape((-1, gd))

        interpolant = NearestNDInterpolator(coords, vp)

        # Get current coordinates
        W = firedrake.VectorFunctionSpace(V.ufl_domain(), V.ufl_element())
        coordinates = firedrake.interpolate(V.ufl_domain().coordinates, W)

    # Get velocity model
    vp_model = firedrake.Function(V)
    if file: vp_model.dat.data[:] = interpolant(coordinates.dat.data)


    return _check_units(vp_model)


def _check_units(c):
    if min(c.dat.data[:]) > 1000.0:
        # data is in m/s but must be in km/s
        if firedrake.COMM_WORLD.rank == 0:
            print("INFO: converting from m/s to km/s", flush=True)
        c.assign(c / 1000.0)  # meters to kilometers
    return c

def spatial_scatter(comm, xi, u):
    """Scatter xi through processes"""

    # Spatial communicator rank and size
    rank = comm.comm.rank
    size = comm.comm.size

    # Update control xi from rank 0
    xi = COMM_WORLD.bcast(xi, root=0)

    # Update Function u
    n = len(u.dat.data[:])
    N = [comm.comm.bcast(n, r) for r in range(size)]
    indices = np.insert(np.cumsum(N), 0, 0)
    u.dat.data[:] = xi[indices[rank] : indices[rank + 1]]

def create_mesh(model, comm, quad=True, diagonal='crossed'):
    """Create mesh from model parameters"""

    origin = tuple(model["mesh"]["origin"]) if "origin" in model["mesh"] else (0, 0) 

    mesh = RectangleMesh(model["mesh"]["nz"],
                                   model["mesh"]["nx"],
                                   model["mesh"]["Lz"],
                                   model["mesh"]["Lx"],
                                   quadrilateral=quad,
                                   diagonal=diagonal,
                                   comm=comm.comm)


    mesh.coordinates.dat.data[:,0] += origin[0]
    mesh.coordinates.dat.data[:,1] += origin[1]
    
    element = FiniteElement(model["opts"]["method"],
                                      mesh.ufl_cell(),
                                      degree=model["opts"]["degree"],
                                      variant=model["opts"]["variant"])

    V = FunctionSpace(mesh, element)

     
    if comm.comm.rank == 0 and comm.ensemble_comm.rank == 0 and "acquisition" in model:
        if "num_sources" in model["acquisition"]:
            print(
                "INFO: Distributing %d shot(s) across %d processor(s). Each shot is using %d cores"
                % (
                    model["acquisition"]["num_sources"],
                    COMM_WORLD.size,
                    COMM_WORLD.size / comm.ensemble_comm.size,
                ),
                flush=True,
            )
    print(
        "  rank %d on ensemble %d owns %d elements and can access %d vertices"
        % (
            mesh.comm.rank,
            comm.ensemble_comm.rank,
            mesh.num_cells(),
            mesh.num_vertices(),
        ),
        flush=True,
    )

    return mesh, V

def legacy_water_layer(mesh, V, vp, depth=None, vw=1.51):
    """Get DoFs in water"""

    # import IPython; IPython.embed()
    if depth:
        z = mesh.coordinates[0]
        water = Function(V).interpolate(conditional(z > depth, 1, 0))
        water_dofs = np.where(water.dat.data[:] > 0.5)
    else:
        water_dofs = np.where(vp.dat.data[:] < vw)

    return water_dofs

def water_layer(mesh, V, vp, model):
    """Get DoFs in water"""

    water_depth = np.array([], dtype=np.int32)
    water_top = np.array([], dtype=np.int32)
    water_bottom = np.array([], dtype=np.int32)
    water_dofs = np.array([], dtype=np.int32)
    if "depth" in model["water"]:
        depth = model["water"]["depth"]
        z = mesh.coordinates[0]
        wata = Function(V).interpolate(conditional(z > depth, 1, 0))
        water_depth = np.where(wata.dat.data[:] > 0.5)
    if "depth_top" in model["water"]:
        depth = model["water"]["depth_top"]
        z = mesh.coordinates[0]
        wata = Function(V).interpolate(conditional(z > depth, 1, 0))
        water_top = np.where(wata.dat.data[:] > 0.5)
    if "depth_bottom" in model["water"]:
        depth = model["water"]["depth_bottom"]
        z = mesh.coordinates[0]
        wata = Function(V).interpolate(conditional(z > depth, 1, 0))
        water_bottom = np.where(wata.dat.data[:] <= 0.5)

    if water_depth or water_top or water_bottom:
        water_dofs = np.union1d(
            np.union1d(water_depth, water_top), water_bottom
        )

    if "by_value" in model["water"]:
        water_dofs = np.where(vp.dat.data[:] < model["water"]["vw"])

    # import IPython; IPython.embed()
    negative_water_dofs = np.setdiff1d(np.array(range(vp.dat.data.size), dtype=np.int64), water_dofs)

    return water_dofs, negative_water_dofs

def material_model(vp, control, model, p=1, salt_p=1):
    """Material Model interpolating function"""

    if "material" in model:

        # Gather values
        vp_min = Constant(model["material"].get("vp_min"))
        vp_max = Constant(model["material"].get("vp_max"))
        vp_salt = Constant(model["material"].get("vp_salt"))

        print(f"salt penalization exponent = {salt_p.dat.data}")

        if model["material"].get("type") in ["integer-mixed", "TOBS", "SIMP"]:

            amin = model["cplex"].get("amin", 0)
            amax = model["cplex"].get("amax", 1)

            # Get fields
            control_salt = control.get("salt")
            control_background = control.get("background")

            # vp.assign(vp_salt * control_salt ** salt_p + _SIMP(control_background, vp_min, vp_max, p=p) * (1 - control_salt ** salt_p))
            vp.assign(
                1 / (Constant(amax) ** salt_p - Constant(amin) ** salt_p) * (
                    vp_salt * (control_salt ** salt_p - Constant(amin) ** salt_p)
                    + _SIMP(control_background, vp_min, vp_max, p=p) * (Constant(amax) ** salt_p - control_salt ** salt_p)
                )
            )


def _SIMP(control, a_min, a_max, p=1):
    """Solid Isotropic Material with Penalization (SIMP)"""

    return a_min + (a_max - a_min) * control ** p

def _dSIMP(dvp, control_background, model, p=1):
    """Solid Isotropic Material with Penalization (SIMP)"""

    # Gather values
    vp_min = Constant(model["material"].get("vp_min"))
    vp_max = Constant(model["material"].get("vp_max"))
    vp_salt = Constant(model["material"].get("vp_salt"))

    dvp.assign(p * (vp_max - vp_min) * control_background ** (p - 1))

def signal_handler(sig, frame):
    save_graphs()
    sys.exit(0)


