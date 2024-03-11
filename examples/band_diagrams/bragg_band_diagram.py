import sys
sys.path.append('..')
sys.path.append('.')

from argparse import ArgumentParser
parser = ArgumentParser("Computes band diagram for Bragg Mirror like in Molding The Flow Of Light")
parser.add_argument("-pw", nargs="+", default=(2, 2), type=int)
parser.add_argument("-eps1", type=float, default=13)
parser.add_argument("-eps2", type=float, default=1)
parser.add_argument("-d1", type=float, default=0.2)
parser.add_argument("-d2", type=float, default=0.8)
parser.add_argument("-a", type=float, default=100)
parser.add_argument("-graph", action="store_true")
parser.add_argument("-compute", action="store_true")
args = parser.parse_args()

import pickle

from bast.tools import nanometers
from bast.tmat.matrices import multS
from bast.tmat.lattice import CartesianLattice
from bast.tmat.scattering import scattering_matrix
from bast.eigentricks import scattering_splitlr
from tqdm import tqdm
import numpy as np
from itertools import product
import scipy
import os

pw = args.pw
assert len(pw) == 2
nx, ny = pw[0] // 2, pw[1] // 2
a  = nanometers(args.a)
lattice = CartesianLattice(pw, a1=(a, 0.0), a2=(0.0, a), eps_emerg=1.0)
N = 100
M = 100
wavelengths = 1 / np.linspace(1./nanometers(10000), 1./nanometers(100), N)
#wavelengths = np.linspace(nanometers(100), nanometers(10000), N)
kplanar = np.linspace(0.0, 3.5 * 2*np.pi / a, M)

# Compute S-matrices
if args.compute:
    Ss = []
    for wl, ky in tqdm(list(product(wavelengths, kplanar))):
        S0, bf = scattering_matrix(pw, lattice, "uniform", [], args.eps1, args.eps1, wl, (0, ky), args.d1*a, 2)
        S1, bf = scattering_matrix(pw, lattice, "uniform", [], args.eps2, args.eps2, wl, (0, ky), args.d2*a, 2)
        S = multS(S0, S1)
        Ss.append((S, wl, ky))

    # Compute the eigenvalues
    points = []
    for S, wl, ky in tqdm(Ss):
        Sl, Sr = scattering_splitlr(S)
        w, v = scipy.linalg.eig(Sl, Sr)
        # On shell condition
        mask = np.logical_not(np.abs(np.abs(w) - 1.0) < 1e-10)
        w[mask] = np.nan
        mode = []
        for i in range(mask.shape[0]):
            evec = v[:, i]
            tetm = abs(evec[0]) > abs(evec[1])
            mode.append(tetm)
        mode = np.asarray(mode)
        points.append((np.angle(w) / 2 / np.pi, np.ones(len(w)) * ky * a / 2 / np.pi, a * np.ones(len(w)) / wl, mode) )
    points = np.asarray(points)
    os.makedirs("./tmp/", exist_ok=True)
    with open("./tmp/bragg_band_diagram.tmp", "wb") as f:
        pickle.dump(points, f)

if args.graph:
    import matplotlib.pyplot as plt
    points = pickle.load(open("./tmp/bragg_band_diagram.tmp", "rb"))
    fig = plt.figure()
    ax = plt.axes(projection='3d', proj_type="ortho")
    mask = np.logical_and(np.logical_and(np.logical_or(np.abs(points[:, 0, :]) < 10.1, np.abs(points[:, 0, :]) > 0.0), points[:, 0, :] >= 0.0) , points[:, 3,:] == 1)
    ax.scatter3D(points[:, 0, :][mask], points[:, 1, :][mask],  points[:, 2, :][mask], marker="x", s=6, alpha=0.5)
    # mask = np.logical_and(np.logical_and(np.abs(points[:, 0, :]) < 0.1, points[:, 0, :] >= 0.0) , points[:, 3,:] == 0)
    # ax.scatter3D(points[:, 0, :][mask], points[:, 1, :][mask],  points[:, 2, :][mask], marker=".", c='r', s=2, alpha=0.5)
    ax.view_init(0, 0)
    plt.show()
