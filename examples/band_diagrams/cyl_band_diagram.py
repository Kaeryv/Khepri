import sys
sys.path.append('..')
sys.path.append('.')

from argparse import ArgumentParser
parser = ArgumentParser("Computes band diagram for Cylinder Array like in Molding The Flow Of Light")
parser.add_argument("-pw", nargs="+", default=(2, 2), type=int)
parser.add_argument("-eps_disk", type=float, default=8.9)
parser.add_argument("-eps_host", type=float, default=1)
parser.add_argument("-a", type=float, default=100)
parser.add_argument("-graph", action="store_true")
parser.add_argument("-compute", action="store_true")
parser.add_argument("-name", default="cyl")
args = parser.parse_args()

import pickle

from khepri.tools import nanometers
from khepri.tmat.matrices import multS
from khepri.tmat.lattice import CartesianLattice
from khepri.tmat.scattering import scattering_matrix
from khepri.eigentricks import scattering_splitlr
from tqdm import tqdm
import numpy as np
from itertools import product
import scipy
import os
import matplotlib.pyplot as plt

pw = args.pw
assert len(pw) == 2
nx, ny = pw[0] // 2, pw[1] // 2
a  = nanometers(args.a)
ax = a
ay = a
az = a / 4
lattice = CartesianLattice(pw, a1=(a, 0.0), a2=(0.0, a), eps_emerg=1.0)
N = 300
M = 30
wavelengths = 1 / np.linspace(1./100/a, 1./a, N)
kplanar = np.linspace(0.0, np.pi / a, M)

# Compute S-matrices
if args.compute:
    Ss = []
    for wl, ky in tqdm(list(product(wavelengths, kplanar))):
        l = 0.5
        S0, bf = scattering_matrix(pw, lattice, "disc", [ l* ax, l * ay, 0.2*ax], args.eps_disk, args.eps_host, wl, (0, ky), az, 3)
        Ss.append((S0, wl, ky))
    # Compute the eigenvalues
    points = []
    for S, wl, ky in tqdm(Ss):
        Sl, Sr = scattering_splitlr(S)
        w, v = scipy.linalg.eig(Sl, Sr)
        # On shell condition
        mask = np.logical_not(np.abs(np.abs(w) - 1.0) < 1e-10)
        w[mask] = np.nan
        mode = []
        Vi = lattice.Vi(wl, (0, ky))
        for i in range(mask.shape[0]):
            evec = v[:, i]
            evec = Vi @ evec
            ng = evec.shape[0] // 4
            tetm = (np.max(np.abs(evec[2*ng:3*ng])) > np.max(np.abs(evec[3*ng:4*ng])))
            mode.append(tetm)
        mode = np.asarray(mode)
        points.append(( a * np.angle(w)/ az /  2 / np.pi, np.ones(len(w)) * ky * ay / 2 / np.pi, a * np.ones(len(w)) / wl, mode))
    points = np.asarray(points)
    os.makedirs("./tmp/", exist_ok=True)
    with open("./tmp/bragg_band_diagram.tmp", "wb") as f:
        pickle.dump(points, f)

if args.graph:
    import matplotlib.pyplot as plt
    points = pickle.load(open("./tmp/bragg_band_diagram.tmp", "rb"))
    fig = plt.figure()
    ax = plt.axes(projection='3d', proj_type="ortho")
    mask = np.logical_and(np.logical_and(np.abs(points[:, 0, :]) < 0.2, points[:, 0, :] >= 0.0), points[:, 3, :] == 0)
    ax.scatter3D(points[:, 0, :][mask], points[:, 1, :][mask],  points[:, 2, :][mask], marker='.', s=1, c='r', alpha=1)
    mask = np.logical_and(np.logical_and(np.abs(points[:, 0, :]) < 0.2, points[:, 0, :] >= 0.0), points[:, 3, :] == 1)
    ax.scatter3D(points[:, 0, :][mask], points[:, 1, :][mask],  points[:, 2, :][mask], marker='.', s=1, c='b', alpha=1)
    #ax.scatter3D(points[:, 0, :], points[:, 1, :],  points[:, 2, :], 'r.', s=2, c='r', alpha=1)
    ax.view_init(0, 0)
    plt.show()
