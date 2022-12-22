from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-angle", type=str, default="0:45:45", help="Angles to compute [degrees].")
parser.add_argument("-freq", type=str, default="0.7:0.84:80", help="Reduced frequencies to compute [c/a]")
parser.add_argument("-range", type=int, nargs="+", required=True, help="Computation range: start, end")
parser.add_argument("-pw", nargs="+", type=int, required=True, help="Plane waves [x,y]")
parser.add_argument("-workdir", type=str, required=True)
parser.add_argument("-collect", action="store_true")
parser.add_argument("-logname", type=str, required=True)
parser.add_argument("-interlayer_depth", default=0.3, type=float, help="Interlayer depth [a]")
parser.add_argument("-a", default=1e-7, type=float, help="Lattice constant [m]")
parser.add_argument("-eps_emergent", default=1.0, type=float, help="Emergent Epsilon")
parser.add_argument("-eps_incident", default=1.0, type=float, help="Incident Epsilon")
parser.add_argument("-eps_interlayer", default=1.0, type=float, help="Interlayer Epsilon")
parser.add_argument("-name", type=str, required=True, help="Data directory [with compute_fan results]")
args = parser.parse_args()

assert args.eps_interlayer == 1.0

import os
os.makedirs(args.workdir, exist_ok=True)

from bast.misc import str2linspace_args as linspace_args
from bast.matrices import multS
from bast.lattice import CartesianLattice
from bast.tools import incident, c, compute_currents
from bast.scattering import scattering_air_tmp

from itertools import product
import numpy as np

amin, amax, ares = linspace_args(args.angle)
fmin, fmax, fres = linspace_args(args.freq)
M, N = ares, fres
pw = args.pw
a = args.a
angles = np.linspace(amin, amax, M)
freqs = np.linspace(fmin, fmax, N) * c / a

def get_l(angle, pw):
    l1 = CartesianLattice(pw, (a, 0), (0., a), args.eps_incident, args.eps_emergent, dtype=np.float64)
    l2 = CartesianLattice(pw, (a, 0), (0., a), args.eps_incident, args.eps_emergent, dtype=np.float64)
    l2.rotate(angle)
    l = l1 + l2
    return l

inputs = np.asarray(list(product(freqs, angles)))

from bast.tools import joint_subspace

# incident wave with polarization 1+1j
pin = incident((pw[0]**2, pw[1]**2), E0=1./np.sqrt(2), p_pol=1, s_pol=0)
from bast.scattering import scattering_matrix
import shutil
transmission = np.zeros((len(list(range(*args.range)))))
for i in range(*args.range):
    print(f"@ iteration {i}")
    freq, angle = inputs[i, :]
    wl = c / freq
    l = get_l(angle, pw=pw)
    Ss = np.load(f"ds/{args.name}/Ss_{i:05d}.npz")
    #Si = np.load(f"ds/interlayers/interlayer_3x3/Ss_{i:05d}.npz")["Si"]
    #Si, _ = scattering_matrix(l.pw, l, "uniform", [], 1.0, 1.0, wl, (0,0), depth=args.interlayer_depth*a)
    Si = scattering_air_tmp(l.gx.shape, l, wl, args.interlayer_depth*a)
    S1 = joint_subspace(Ss["S1s"], kind=0)
    S2 = joint_subspace(Ss["S2s"], kind=1)
    # S1 @ Si @ S2
    Stot = multS(S1, Si)
    Stot = multS(Stot, S2)
    j1plus, j1minus, j3plus = compute_currents(pin, Stot @ pin, l, wl)
    Ttot= np.abs(np.sum(j3plus) / np.sum(j1plus))
    transmission[i-args.range[0]] = Ttot

    os.remove(f"ds/{args.name}/Ss_{i:05d}.npz")

np.save(f"{args.workdir}/T_{args.logname}.npy", transmission)
print("Done computing transmission.")
