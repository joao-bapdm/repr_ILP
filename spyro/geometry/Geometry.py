from firedrake import *

import spyro

class Geometry:

    def __init__(self, model, mesh, V, comm):
        """Class describing acquisition geometry.
    
        Parameters
        ----------
        model: `dictionary`
            Contains simulation parameters and options.
        mesh: a Firedrake.mesh
            2D/3D simplicial mesh read in by Firedrake.Mesh
        V: Firedrake.FunctionSpace object
            The space of the finite elements
        comm: Firedrake.ensemble_communicator
            An ensemble communicator
        """

        self.V = V
        self.mesh = mesh
        self.my_ensemble = comm
        acq = model["acquisition"]
        if "src_depth" in acq:
            self.src_depth = model["acquisition"]["src_depth"]
        if "num_sources" in acq:
            self.num_sources = model["acquisition"]["num_sources"]
        if "src_XMIN" in acq:
            self.src_XMIN = model["acquisition"]["src_XMIN"]
        if "src_XMAX" in acq:
            self.src_XMAX = model["acquisition"]["src_XMAX"]
        if "rec_depth" in acq:
            self.rec_depth = model["acquisition"]["rec_depth"]
        if "num_receivers" in acq:
            self.num_receivers = model["acquisition"]["num_receivers"]
        if "rec_XMIN" in acq:
            self.rec_XMIN = model["acquisition"]["rec_XMIN"]
        if "rec_XMAX" in acq:
            self.rec_XMAX = model["acquisition"]["rec_XMAX"]
        self.model = model

    def create_sources(self):
        """Create sources transect"""
    
        # self.model["acquisition"]["source_pos"] = spyro.create_receiver_transect(
        #     (self.src_depth, self.src_XMIN), (self.src_depth, self.src_XMAX), self.num_sources
        # )

        #if not self.model["acquisition"]["source_pos"]:
        if "source_pos" not in self.model["acquisition"]:

            self.model["acquisition"]["source_pos"] = []

            for depth in self.src_depth:
                self.model["acquisition"]["source_pos"] += spyro.create_receiver_transect(
                    (depth, self.src_XMIN), (depth, self.src_XMAX), self.num_sources
                ).tolist()
            self.model["acquisition"]["source_pos"] = np.array(self.model["acquisition"]["source_pos"])

        self.model["acquisition"]["num_sources"] = len(self.model["acquisition"]["source_pos"])
        sources = spyro.Sources(self.model, self.mesh, self.V, self.my_ensemble).create()
    
        return sources
    
    def create_receivers(self):
        "Create receivers transect"""

        # if not self.model["acquisition"]["receiver_locations"]: 
        if "receiver_locations" not in self.model["acquisition"]: 

            self.model["acquisition"]["receiver_locations"] = []

            for depth in self.rec_depth:
                self.model["acquisition"]["receiver_locations"] += spyro.create_receiver_transect(
                    (depth, self.rec_XMIN), (depth, self.rec_XMAX), self.num_receivers
                ).tolist()
            self.model["acquisition"]["receiver_locations"] = np.array(self.model["acquisition"]["receiver_locations"])
            self.model["acquisition"]["num_receivers"] = len(self.model["acquisition"]["receiver_locations"])
    
        self.model["acquisition"]["num_receivers"] = len(self.model["acquisition"]["receiver_locations"])
        receivers = spyro.Receivers(self.model, self.mesh, self.V, self.my_ensemble).create()
    
        return receivers

    def create(self):
        """Create both sources and receivers"""
    
        sources = self.create_sources()
        receivers = self.create_receivers()
        # receivers = None
    
        return sources, receivers
