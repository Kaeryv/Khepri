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
parser.add_argument("-name", default="dos_diagram")
args = parser.parse_args()

import pickle

from bast.tools import nanometers
from bast.matrices import multS
from bast.lattice import CartesianLattice
from bast.scattering import scattering_matrix
from bast.eigentricks import scattering_splitlr, scattering_det
from tqdm import tqdm
import numpy as np
from itertools import product
import scipy
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

pw = args.pw
assert len(pw) == 2
a  = nanometers(args.a)
lattice = CartesianLattice(pw, a1=(a/2, 0.0), a2=(0.0, a/2), eps_incid=1.0)
N = 200
M = 50
wavelengths = 1 / np.linspace(1./nanometers(10000), 1./nanometers(100), N)
kplanar = np.linspace(0.0, 2*np.pi / a, M)

# Compute S-matrices
if args.compute:
    Ss = []
    for wl, ky in tqdm(list(product(wavelengths, kplanar))):
        kp = (0, ky)
        S0, bf = scattering_matrix(pw, lattice, "uniform", [], args.eps1, args.eps1, wl, kp, args.d1*a, 2)
        S1, bf = scattering_matrix(pw, lattice, "uniform", [], args.eps2, args.eps2, wl, kp, args.d2*a, 2)
        S = multS(S1, S0)
        Ss.append((S, wl, ky))
    import pickle
    with open(f"tmp/{args.name}.tmp", "wb") as f:
        pickle.dump(Ss, f)

if args.graph:
    import pickle
    with open(f"tmp/{args.name}.tmp", "rb") as f:
        Ss = pickle.load(f)
        
    import matplotlib.pyplot as plt
    dos = []
    for S, wl, ky in tqdm(Ss):
        #Sl, Sr = scattering_splitlr(S)
        #P = Sl - np.exp(1j*0)* Sr
        #det = scattering_det(S)
        det = scipy.linalg.det(S+np.eye(*S.shape))
        dos.append(det)

    dos = np.asarray(dos).reshape(N, M)
    dos = np.flipud(dos)
    dos = (np.gradient(np.log(dos)[:-1,:], 0.1, axis=0)) / 1j / 2 / np.pi
    fig, axs = plt.subplots(1, 4)
    extent=[0, 1, 0, 1]
    for i, field in enumerate([ dos.real, dos.imag, np.abs(dos), np.angle(dos)]):
        h = axs[i].matshow(field, extent=extent)
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(h, cax=cax, orientation='vertical')
    plt.show()
