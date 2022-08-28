import os

import firedrake
from firedrake.petsc import PETSc
import numpy as np
import finat
from ROL.firedrake_vector import FiredrakeVector as FeVector
import ROL
from mpi4py import MPI

import spyro
import spyro.material as material
from spyro.optimizers import adam, tobs

outdir = "fwi_ball_tobs/"
model = spyro.io.load_model()

class Model():
    def __init__(self, model, c=None):
        # load config file and set up communicator
        self.model = model
        self.comm = spyro.utils.mpi_init(model)
        # create mesh
        self.init_regular_mesh()
        # create function space
        self.V = firedrake.FunctionSpace(
            self.mesh,
            self.model["opts"]["method"],
            self.model["opts"]["degree"]
        )
        # set up the control
        self.init_control(c=c)
        # set up acquisition geometry
        self.init_geometry()
        # set up source wavelet
        self.set_wavelet()
        # water, where gradient is zero
        self.water = np.where(self.mesh.coordinates.dat.data[:,0] < self.model["opts"]["water_depth"])

    def set_output(self, fname=""):
        """Create pvd output files for visualizing control and gradient"""
        PETSc.Sys.Print(f"Saving output at {outdir}, base file name is {fname}", comm=firedrake.COMM_WORLD)
        if self.comm.ensemble_comm.rank == 0:
            self.control_file = firedrake.File(outdir + "control" + fname + ".pvd", comm=self.comm.comm)
            self.material_file = firedrake.File(outdir + "material" + fname + ".pvd", comm=self.comm.comm)
            self.grad_file = firedrake.File(outdir + "grad" + fname + ".pvd", comm=self.comm.comm)

    def init_regular_mesh(self):
        """initial regular 2 or 3 mesh"""
        if self.model["opts"]["dimension"] == 3:
            self.mesh = firedrake.CubeMesh(
                nx=self.model["mesh"]["nx"],
                ny=self.model["mesh"]["ny"],
                nz=self.model["mesh"]["nz"],
                L=1,
                distribution_parameters={
                    "overlap_type": (firedrake.DistributedMeshOverlapType.NONE, 0)
                },
                comm=self.comm.comm
            ) 
        elif self.model["opts"]["dimension"] == 2:
            raise NotImplementedError(
                "Bi-dimensional case not implemented"
            )

    def init_control(self, c=None):
        """initiate control variable"""
        self.control = spyro.io.interpolate(
            self.model, self.mesh, self.V, fname=self.model["files"]["input_file"]
        )
        if c is not None:
            self.control.dat.data[:] = c

    def init_geometry(self):
        """set up sources and receivers"""
        self.sources = spyro.Sources(self.model, self.mesh, self.V, self.comm)
        self.receivers = spyro.Receivers(self.model, self.mesh, self.V, self.comm)

    def set_wavelet(self, lowpass=None):
        """set up the wavelet object, optionally low pass filtering the signal.

        Parameters
        ----------
        lowpass: float
            max frequency present in the filtered signal (Hz).
        """
        self.wavelet = spyro.full_ricker_wavelet(
            dt=self.model["timeaxis"]["dt"],
            tf=self.model["timeaxis"]["tf"],
            freq=self.model["acquisition"]["frequency"],
            cutoff=lowpass
        )

class WavePropagator(Model):
    def __init__(self, model, c=None):
        super().__init__(model, c=c)
        self.p_guess = None
        self.p_guess_recv = None
        self.p_exact_recv = None
        self.p_exact_recv_filtered = None
        self.misfit = 0.0
        self.vp = firedrake.Function(self.control.function_space())
        # self.vp.assign(material.model.simp(self.control, model["material"]["vp_min"], model["material"]["vp_max"]))
        self.vp.assign(material.model.interpolate(self.model, self.control))
        self.dJ = firedrake.Function(self.V, name="gradient")

    def load_shots(self, file_name=None):
        self.p_exact_recv = spyro.io.load_shots(self.model, self.comm, file_name=file_name)

    def filter_shot_record(self, lowpass=None):
        """Filter shot record"""
        self.p_exact_recv_filtered = spyro.utils.butter_lowpass_filter(self.p_exact_recv, lowpass, 1.0 / self.model["timeaxis"]["dt"])

    def forward(self):
        """Compute the forward"""
        self.p_guess, self.p_guess_recv = spyro.solvers.forward(
            self.model,
            self.mesh,
            self.comm,
            self.vp,
            self.sources,
            self.wavelet,
            self.receivers,
        )

    def value(self):
        """Compute the objective function"""
        J_total = np.zeros((1))
        self.misfit = spyro.utils.evaluate_misfit(
            self.model, self.p_guess_recv, self.p_exact_recv_filtered
        )
        print("computing the functional")
        J_total[0] += spyro.utils.compute_functional(self.model, self.misfit, velocity=self.vp)
        J_total = firedrake.COMM_WORLD.allreduce(J_total, op=MPI.SUM)
        J_total[0] /= self.comm.ensemble_comm.size
        if self.comm.comm.size > 1:
            J_total[0] /= self.comm.comm.size
        return J_total[0]

    def gradient(self):
        """Compute the gradient of the functional"""
        kernel = spyro.solvers.gradient_switch(
            self.model,
            self.mesh,
            self.comm,
            self.vp,
            self.receivers,
            self.p_guess,
            self.misfit,
        )
        dJ_local = spyro.solvers.deriv(self.model, kernel, 2 * self.vp)
        if self.comm.ensemble_comm.size > 1:
            self.comm.allreduce(dJ_local, self.dJ)
        else:
            self.dJ = dJ_local
        self.dJ /= self.comm.ensemble_comm.size
        if self.comm.comm.size > 1:
            self.dJ /= self.comm.comm.size
        # regularize the gradient if asked.
        if self.model["cplex"]["use_rmin"]: 
            spyro.utils.helmholtz_filter(self.dJ, model['cplex']['rmin'])
        # mask the water layer
        self.dJ.dat.data[self.water] = 0.0
        # Visualize
        if self.comm.ensemble_comm.rank == 0:
            self.grad_file.write(self.dJ)
        return self.dJ 

    def update(self, control):
        """Update the control variable"""
        self.control.dat.data[:] = control
        # self.vp.assign(
        #     material.model.simp(
        #         self.control,
        #         self.model["material"]["vp_min"],
        #         self.model["material"]["vp_max"])
        # )
        self.vp.assign(material.model.interpolate(self.model, self.control))
        # Visualize
        if self.comm.ensemble_comm.rank == 0:
            self.control_file.write(self.control)
            self.material_file.write(self.vp)

Prop = WavePropagator(model, c=0)
Prop.load_shots(file_name=model["shots"])

# cplex parameters
m, v = 0, 0
# initiate controls
xi = Prop.control.dat.data
J, J0, dJ = 0, 0, 0
# initiate log file
os.makedirs(outdir, exist_ok=True)
logfile = os.path.join(outdir, "history.log")
with open(logfile, "w") as f:
    f.write("OPTIMIZATION HISTORY\n")
    f.write("====================\n")

# frequency loop
for freq in model["inversion"]["freq_bands"]:

    # set up output files according to frequency
    fname = str(freq) + "Hz" if freq else ""
    Prop.set_output(fname)

    # filter wavelet and shot record
    Prop.set_wavelet(lowpass=freq)
    Prop.filter_shot_record(lowpass=freq)

    counter = 0
    # optimization loop
    while tobs.check_convergence(
            counter,
            model["inversion"]["max_iter"],
            J,
            J0,
            dJ
        ):
        # update values
        J0 = J

        # get objective function and gradient
        Prop.forward()
        J = Prop.value()
        dJ = Prop.gradient().dat.data

        # get first moment
        m = adam.moving_avg(model["cplex"]["gamma_m"], m, dJ)
        # second moment
        v = adam.moving_avg(model["cplex"]["gamma_v"], v, dJ ** 2)

        # correct for bias
        m_ = adam.remove_bias(model["cplex"]["gamma_m"], m, counter)
        v_ = adam.remove_bias(model["cplex"]["gamma_v"], v, counter)

        # evaluate RMSProp averaged gradient
        dJ = adam.RMSprop(m_, v_)

        # output info to stdout and to file
        tobs.print_iter_info(
            counter,
            J,
            (J - J0) / J0 * 100 if J0 else 0,
            model["cplex"]["beta"],
            model["cplex"]["rmin"],
            logfile=logfile
        )

        # update beta and rmin
        tobs.update_flip_limits(
            model["cplex"]["beta"],
            counter,
            model["cplex"]["mul_beta"],
            xi
        )
        tobs.update_rmin(
            model["cplex"]["rmin"],
            counter,
            model["cplex"]["lim_rmin"],
            model["cplex"]["mul_rmin"]
        )

        # update control
        dxi = tobs.TOBS(dJ,
            np.zeros(dJ.shape), 
            model["cplex"]["gbar"],
            xi.mean(),
            model["cplex"]["epsilons"],
            model["cplex"]["beta"],
            xi)
        Prop.update(xi + dxi)
        counter+=1

