from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-input", type=str, required=True)
parser.add_argument("-angle", type=float, required=True)
args = parser.parse_args()

import os
import json

if os.path.isfile(args.input):
    with open(args.input, "r") as f:
        config = json.load(f)["args"]


import numpy as np
from khepri.tmat.matrices import multS, matrix_s
from khepri.tmat.lattice import CartesianLattice
from khepri.tools import incident, c, compute_currents, rotation_matrix
from math import cos, sin
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided


pw = config["pw"]
a = 1e-7


l1 = CartesianLattice(pw, (a, 0), (0., a), 1.0, 1.0)
l2 = CartesianLattice(pw, (a, 0), (0., a), 1.0, 1.0)
l2.rotate(args.angle)

# We have two lattices, compute the master lattice by adding them
l = l1 + l2
#mask = l.restrict_radius_mask(20e8)


fig, axs = plt.subplots(1,2, figsize=(25, 25))
#l.Gy[np.logical_not(mask.reshape(25,25))] = np.nan
#l.Gx[np.logical_not(mask.reshape(25,25))] = np.nan
axs[0].matshow(l.gx.real, vmin=-2e8, vmax=2e8, cmap="RdBu")
im = axs[1].matshow(l.gy.real, vmin=np.nanmin(l.gy.real), vmax=2e8, cmap="RdBu")
for (i, j), z in np.ndenumerate(l.gy.real):
    axs[1].text(j, i, '{:0.3f}'.format(z*1e-8), ha='center', va='center')
plt.savefig("Figures/Gxy.png")


fig, axs = plt.subplots(figsize=(8, 8))
GS = np.vstack(list(l.g_vectors()))
#GS = GS[mask]
plt.scatter(l.gx.flatten()*a/2/np.pi, l.gy.flatten()*a/2/np.pi)
tester = set()
for x, y in zip(l.gx.flat, l.gy.flat):
    tester.add((x,y))
print(len(tester))
plt.savefig("Figures/ExtendedGrid.png")

fig, axs = plt.subplots(figsize=(8, 8))
GS = np.vstack(list(l1.g_vectors()))
GS2 = np.vstack(list(l2.g_vectors()))
plt.scatter(*GS.T*a/2/np.pi)
plt.scatter(*GS2.T*a/2/np.pi, c='r')
plt.savefig("Figures/OriginalGrid.png")
