"""Generate 3D velocity model with ball at the center"""

import matplotlib.pyplot as plt
import numpy as np
import h5py

L = 1  # Length
nn = 153  # number of nodes per axis

# coordinates
xi = np.linspace(0, 1, nn)
yi = np.linspace(0, 1, nn)
zi = np.linspace(0, 1, nn)
X, Y, Z = np.meshgrid(xi, yi, zi)

# velocity model
vp = 2.0 * np.ones((nn, nn, nn))
vp[(X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2 < 0.15 ** 2] = 3
# vp[Z < 0.25] = 1.6

# save results
with h5py.File("centered_ball.hdf5", "w") as f:
    f.create_dataset("velocity_model", data=vp, dtype="f")

