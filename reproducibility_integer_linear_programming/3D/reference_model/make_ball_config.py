"""Create json file for ball case"""


import json

import numpy as np

nn = 50  # number of elements per axis
L = 1  # Cube size

model = {}

model["opts"] = {
    "method": "KMV",
    "variant": "KMV",
    "degree": 1,
    "dimension": 3,
    "regularization": False
}

model["mesh"] = {
    "nz": nn,
    "nx": nn,
    "ny": nn,
    "Lz": L,
    "Lx": L,
    "Ly": L,
    "meshfile": None,
    "initmodel": None,
    "truemodel": None
}

model["parallelism"] = {"type": "automatic"}

model["timeaxis"] = {
    "t0": 0.0,
    "tf": 2.0,
    "dt": 0.001,
    "nspool": 99999,
    "fspool": 1
}

model["BCs"] = {
    "status": False,
    "outer_bc": None,
}

# source_pos = [[0.5, 0.5, 0.02], [0.4, 0.4, 0.02]]
# source_pos = [[0.5, 0.5, 0.02]]
source_pos = []
sz = 0.02
for sx in np.linspace(0, 1, 3):
    for sy in np.linspace(0, 1, 3):
        source_pos.append([sz, sx, sy])

# receiver_locations = [[0.5, sy, 0.03] for sy in np.linspace(0, 1, 200)]
receiver_locations = []
rz = 0.98
for rx in np.linspace(0, 1, 51):
    for ry in np.linspace(0, 1, 51):
        receiver_locations.append([rz, rx, ry])

# Make acquisition geometry
model["acquisition"] = {
    "source_type": "Ricker",
    "amplitude": 1,
    "frequency": 4,
    "delay": 1.0,
    "source_pos": source_pos,
    "receiver_locations": receiver_locations,
    "num_sources": len(source_pos),
    "num_receivers": len(receiver_locations)
}

with open("centered_ball.json", "w") as f:
    json.dump(model, f, indent=4)

