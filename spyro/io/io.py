from __future__ import with_statement

from io import StringIO
import os
import json
import argparse

import firedrake as fire

from scipy.interpolate import RegularGridInterpolator
import pickle
import numpy as np
import matplotlib.pyplot as plt
import h5py

from .. import domains

__all__ = ["save_shots", "load_shots", "read_mesh", "interpolate"]

class Callback:

    def __init__(self, model, comm, name=""):
        """Class for writing output.

        Parameters
        ----------
        model: dict
            Contains simulation parameters and options.
        comm: Firedrake.ensemble_communicator
            An ensemble communicator
        """

        self.comm = comm
        self.model = model
        self.name = name

    def create_file(self, m=None, dm=None, vp=None):
        """Create output file(s)"""

        outdir = self.model["output"]["outdir"]
        os.makedirs(outdir, exist_ok=True)
        mfile = outdir+"/""m"+self.name+".pvd"
        dmfile = outdir+"/""dm"+self.name+".pvd"
        vpfile = outdir+"/""vp"+self.name+".pvd"

        # if self.comm.ensemble_comm.rank == 0:
        if True:
            if m:
                self.m_file = fire.File(
                    mfile, comm=self.comm.comm
                )
                self.m_file.write(m)
            if dm:
                self.dm_file = fire.File(
                    dmfile, comm=self.comm.comm
                )
                self.dm_file.write(dm)
            if vp:
                self.vp_file = fire.File(
                    vpfile, comm=self.comm.comm
                )
                self.vp_file.write(vp)

    def write_file(self, m=None, dm=None, vp=None):
        "Write output file(s)"""

        # if self.comm.ensemble_comm.rank == 0:
        if True:
            if m:
                self.m_file.write(m)
            if dm:
                self.dm_file.write(dm)
            if vp:
                self.vp_file.write(vp)

def save_image(field, fname=None, cmap="seismic", format=None, subplots=None):
    """save firedrake.Function as imagefile"""
    fname = field.name() if not fname else fname
    fig, axes = plt.subplots() if not subplots else subplots
    fire.tripcolor(field, axes=axes, cmap=cmap); axes.set_aspect("equal")
    fig.savefig(fname, format=format)
    plt.close(plt.gcf())

def save_shots(filename, array):
    """Save a `numpy.ndarray` to a `pickle`.

    Parameters
    ----------
    filename: str
        The filename to save the data as a `pickle`
    array: `numpy.ndarray`
        The data to save a pickle (e.g., a shot)

    Returns
    -------
    None

    """
    with open(filename, "wb") as f:
        pickle.dump(array, f)
    return None


def load_shots(filename):
    """Load a `pickle` to a `numpy.ndarray`.

    Parameters
    ----------
    filename: str
        The filename to save the data as a `pickle`

    Returns
    -------
    array: `numpy.ndarray`
        The data

    """

    with open(filename, "rb") as f:
        array = np.asarray(pickle.load(f), dtype=float)
    return array

def load_shot(filename, num_shots, ens_comm):
    """Load a 'pickle' to a 'numpy.ndarray',

    Parameters
    ----------
    filename: str
        The base filename
    sn: int
        shot record number
    ens_comm: Comunnicator
        Ensemble communicator

    Returns
    -------
    shots: 'numpy.ndarray'
        The data
    """
    shots = []
    n_loops = num_shots // ens_comm.size + 1
    for i in range(n_loops):
        # locate shot number
        sn = ens_comm.rank + i * ens_comm.size
        # check shot limit
        if sn < num_shots:
            # assign corect name
            shot_filename = filename + "_sn_" + str(sn) + ".dat"
            # append to shot record list
            shots.append(load_shots(shot_filename))
        # else:
        #     shots.append(np.array([]))

    return shots
                

def is_owner(ens_comm, rank):
    """Distribute shots between processors in using a modulus operator

    Parameters
    ----------
    ens_comm: Firedrake.ensemble_communicator
        An ensemble communicator
    rank: int
        The rank of the core

    Returns
    -------
    boolean
        `True` if `rank` owns this shot

    """
    return ens_comm.ensemble_comm.rank == (rank % ens_comm.ensemble_comm.size)


def _check_units(c):
    if min(c.dat.data[:]) > 100.0:
        # data is in m/s but must be in km/s
        if fire.COMM_WORLD.rank == 0:
            print("INFO: converting from m/s to km/s", flush=True)
        c.assign(c / 1000.0)  # meters to kilometers
    return c


def interpolate(model, mesh, V, guess=False):
    """Read and interpolate a seismic velocity model stored
    in a HDF5 file onto the nodes of a finite element space.

    Parameters
    ----------
    model: `dictionary`
        Model options and parameters.
    mesh: Firedrake.mesh object
        A mesh object read in by Firedrake.
    V: Firedrake.FunctionSpace object
        The space of the finite elements.
    guess: boolean, optinal
        Is it a guess model or a `exact` model?

    Returns
    -------
    c: Firedrake.Function
        P-wave seismic velocity interpolated onto the nodes of the finite elements.

    """
    sd = V.mesh().geometric_dimension()
    m = V.ufl_domain()
    if model["PML"]["status"]:
        minz = -model["mesh"]["Lz"] - model["PML"]["lz"]
        maxz = 0.0
        minx = 0.0 - model["PML"]["lx"]
        maxx = model["mesh"]["Lx"] + model["PML"]["lx"]
        miny = 0.0 - model["PML"]["ly"]
        maxy = model["mesh"]["Ly"] + model["PML"]["ly"]
    else:
        pad = model["mesh"].get("pad", 0)
        minz = -model["mesh"]["Lz"] - pad
        maxz = 0.0
        minx = 0.0 - pad
        maxx = model["mesh"]["Lx"] + pad
        miny = 0.0 - pad
        maxy = model["mesh"]["Ly"] + pad
        
    W = fire.VectorFunctionSpace(m, V.ufl_element())
    coords = fire.interpolate(m.coordinates, W)
    # (z,x) or (z,x,y)
    if sd == 2:
        qp_z, qp_x = coords.dat.data[:, 0], coords.dat.data[:, 1]
    elif sd == 3:
        qp_z, qp_x, qp_y = (
            coords.dat.data[:, 0],
            coords.dat.data[:, 1],
            coords.dat.data[:, 2],
        )
    else:
        raise NotImplementedError

    if guess:
        fname = model["mesh"]["initmodel"]
    else:
        fname = model["mesh"]["truemodel"]

    with h5py.File(fname, "r") as f:
        Z = np.asarray(f.get("velocity_model")[()])

        if sd == 2:
            nrow, ncol = Z.shape
            z = np.linspace(minz, maxz, nrow)
            x = np.linspace(minx, maxx, ncol)

            # make sure no out-of-bounds
            qp_z2 = [minz if z < minz else maxz if z > maxz else z for z in qp_z]
            qp_x2 = [minx if x < minx else maxx if x > maxx else x for x in qp_x]

            interpolant = RegularGridInterpolator((z, x), Z)
            tmp = interpolant((qp_z2, qp_x2))
        elif sd == 3:
            nrow, ncol, ncol2 = Z.shape
            z = np.linspace(minz, maxz, nrow)
            x = np.linspace(minx, maxx, ncol)
            y = np.linspace(miny, maxy, ncol2)

            # make sure no out-of-bounds
            qp_z2 = [minz if z < minz else maxz if z > maxz else z for z in qp_z]
            qp_x2 = [minx if x < minx else maxx if x > maxx else x for x in qp_x]
            qp_y2 = [miny if y < miny else maxy if y > maxy else y for y in qp_y]

            interpolant = RegularGridInterpolator((z, x, y), Z)
            tmp = interpolant((qp_z2, qp_x2, qp_y2))

    c = fire.Function(V)
    c.dat.data[:] = tmp
    c = _check_units(c)
    return c


def read_mesh(model, ens_comm):
    """Reads in an external mesh and scatters it between cores.

    Parameters
    ----------
    model: `dictionary`
        Model options and parameters.
    ens_comm: Firedrake.ensemble_communicator
        An ensemble communicator

    Returns
    -------
    mesh: Firedrake.Mesh object
        The distributed mesh across `ens_comm`.
    V: Firedrake.FunctionSpace object
        The space of the finite elements

    """

    method = model["opts"]["method"]
    degree = model["opts"]["degree"]

    num_sources = model["acquisition"]["num_sources"]
    mshname = model["mesh"]["meshfile"]

    if method == "CG" or method == "KMV":
        mesh = fire.Mesh(mshname, comm=ens_comm.comm, distribution_parameters={"overlap_type": (fire.DistributedMeshOverlapType.NONE,0)})
    else:
        mesh = fire.Mesh(mshname, comm=ens_comm.comm)
    if ens_comm.comm.rank == 0 and ens_comm.ensemble_comm.rank == 0:
        print(
            "INFO: Distributing %d shot(s) across %d processor(s). Each shot is using %d cores"
            % (
                num_sources,
                fire.COMM_WORLD.size,
                fire.COMM_WORLD.size / ens_comm.ensemble_comm.size,
            ),
            flush=True,
        )
    print(
        "  rank %d on ensemble %d owns %d elements and can access %d vertices"
        % (
            mesh.comm.rank,
            ens_comm.ensemble_comm.rank,
            mesh.num_cells(),
            mesh.num_vertices(),
        ),
        flush=True,
    )
    # Element type
    element = domains.space.FE_method(mesh, method, degree)
    # Space of problem
    return mesh, fire.FunctionSpace(mesh, element)

def load_model(jsonfile=None):
    """Load model dictionary describing forward/inversion problem"""

    parser = argparse.ArgumentParser(description="Run Full Waveform Inversion")
    parser.add_argument(
        "-c", "--config-file",type=str, required=False, help="json file with parameters"
    )
    parser.add_argument(
        "-i", "--input-field",type=str, required=False, help="hdf5 file with initial guess"
    )
    parser.add_argument(
        "-s", "--salt-field",type=str, required=False, help="hdf5 file with initial salt guess"
    )
    parser.add_argument(
        "-b", "--background-field",type=str, required=False, help="hdf5 file with initial background guess"
    )
    parser.add_argument(
        "-e", "--exact-field",type=str, required=False, help="hdf5 file with exact field"
    )
    parser.add_argument(
        "-o", "--output-field",type=str, required=False, help="hdf5 file where result is stored"
    )
    parser.add_argument(
        "-O", "--optimizer",type=str, required=False, help="type of optimizer used"
    )
    parser.add_argument(
        "--control", action="store_true", help="treat input as control variable instead of velocity field"
    )

    file = parser.parse_args().config_file if not jsonfile else jsonfile
    inputfile = parser.parse_args().input_field
    saltfile = parser.parse_args().salt_field
    backgroundfile = parser.parse_args().background_field
    outputfile = parser.parse_args().output_field
    exactfile = parser.parse_args().exact_field
    optimizer = parser.parse_args().optimizer

    with open(file, "r") if file else StringIO('{}') as f:
        model = json.load(f)

    if parser.parse_args().control: model["opts"]["control"] = True

    if inputfile:
        if "data" in model:
            model["data"]["initfile"] = inputfile
        else:
            model["data"] = {"initfile": inputfile}
    else:
        if "data" in model:
            model["data"]["initfile"] = None
        else:
            model["data"] = {"initfile": None}
    if saltfile:
        if "data" in model:
            model["data"]["saltfile"] = saltfile
        else:
            model["data"] = {"saltfile": saltfile}
    else:
        if "data" in model:
            model["data"]["saltfile"] = None
        else:
            model["data"] = {"saltfile": None}
    if backgroundfile:
        if "data" in model:
            model["data"]["backgroundfile"] = backgroundfile
        else:
            model["data"] = {"backgroundfile": backgroundfile}
    else:
        if "data" in model:
            model["data"]["backgroundfile"] = None
        else:
            model["data"] = {"backgroundfile": None}
    if exactfile:
        if "data" in model:
            model["data"]["exactfile"] = exactfile
        else:
            model["data"] = {"exactfile": exactfile}
    else:
       if "data" in model:
           model["data"]["exactfile"] = None
       else:
           model["data"] = {"exactfile": None}
    if outputfile:
        if "data" in model:
            model["data"]["resultfile"] = outputfile
        else:
            model["data"] = {"resultfile": outputfile}
    if optimizer:
        if "inversion" in model:
            model["inversion"]["optimizer"] = optimizer
        else:
            model["inversion"] = {"optimizer": optimizer}
    else:
        if "inversion" in model:
            if "optimizer" in model["inversion"].keys():
                pass
            else:
                model["inversion"]["optimizer"] = None
        else:
            model["inversion"] = {"optimizer": None}
    
    # set output naming
    if "output" not in model:
        dest = os.path.join("results", file.split("/")[-1])
        model["output"] = {}
    if "outdir" not in model["output"]:
        model["output"]["outdir"] = dest.rstrip(".json")

    if "data" not in model:
        model["data"] = {}

    # for saving config file
    model["data"]["configfile"] = file

    for key, ext in zip(["pic", "resultfile", "fobj"], [".png", ".hdf5", ".npy"]):
        if key not in model["data"]:
            model["data"][key] = file.replace(".json", ext).split("/")[-1]

    if "shots" not in model["data"]:
        model["data"]["shots"] = os.path.join("shots", file.rstrip(".json").split("/")[-1])

    return model

def save_model(model, jsonfile=None):
    """Save model dictionary describing forward/inversion problem"""

    if not jsonfile: jsonfile = "model.json"

    with open(jsonfile, "w") as f:
        json.dump(model, f, indent=4)

def log_new_frequency(freq, log_file):
    """Simply print freq info to log file"""
    # freq_band = "" if not freq else freq
    if not freq: return
    with open(log_file, "a") as f:
        # if comm.comm.rank == 0 and comm.ensemble_comm.rank == 0:
        f.write("\n"+"--------------------")
        f.write(f"inverting for lowpassed {freq}Hz signal")
        f.write("--------------------")

    
