'''
'''

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-interlayer_depth", default=0.3, type=float, help="Interlayer depth [a]")
parser.add_argument("-freq", type=str, default="0.7:0.84:80", help="Reduced frequencies to compute [c/a]")
parser.add_argument("-pw", nargs="+", type=int, required=True, help="Plane waves [x,y]")
parser.add_argument("-output", type=str, required=True, help="Output directory [empty folder]")
parser.add_argument("-a", default=1e-7, type=float, help="Lattice constant [m]")
parser.add_argument("-hole_radius", default=0.25, type=float, help="Holes Radius [a]")
parser.add_argument("-hole_position", default=[0.5, 0.5], type=float, nargs="+", help="Holes Position x y [a]")
parser.add_argument("-layers_depth", default=0.55, type=float, help="Layers depth [a]")
parser.add_argument("-eps_emergent", default=1.0, type=float, help="Emergent Epsilon")
parser.add_argument("-eps_interlayer", default=1.0, type=float, help="Interlayer Epsilon")
parser.add_argument("-eps_incident", default=1.0, type=float, help="Incident Epsilon")
parser.add_argument("-eps_layer", default=12.0, type=float, help="Layer Epsilon")
parser.add_argument("-eps_holes", default=1.0, type=float, help="Holes Epsilon")
parser.add_argument("-interlayer-shift", default=None, type=float, help="Shift between unit cells.")
args = parser.parse_args()

assert len(args.pw) == 2
assert args.a > 0.0
from bast.tools import incident, c, compute_currents
from bast.matrices import multS
# We need the twisted and untwisted lattices
from bast.lattice import CartesianLattice
import numpy as np
import os

pw = args.pw
a = args.a
r = args.hole_radius
x, y = args.hole_position
disk_params = [ x*a, y*a, r*a] # Disk parameters [m]

eps_emergent = args.eps_emergent
eps_incident = args.eps_incident

from bast.scattering import scattering_air_tmp, scattering_interface, scattering_matrix
from bast.tools import c
from  bast.matrices import matrix_s

l = CartesianLattice(pw, (a, 0), (0., a), eps_incident, eps_emergent, dtype=np.float32)

# Let's get all the alpha, wl pairs in a list.
import bast.misc as bm
from itertools import product
f = np.linspace(*bm.str2linspace_args(args.freq)) * c / a
wavelength = c / f
pin = incident(pw, E0=1.0 , p_pol=1.0, s_pol=1.0j) # 1j, 1 /np.sqrt(2)
print(np.min(f), np.max(f))
transmission = list()
for i, wl in enumerate(wavelength):
    print(f"At iteration {i} over {len(wavelength)}." )
    
    S, _ = scattering_matrix(pw, l, "disc", disk_params, args.eps_holes, args.eps_layer, wavelength=wl, kp=(0,0), depth=a*args.layers_depth)
    Si, _ = scattering_matrix(pw, l, "uniform", [], args.eps_interlayer, args.eps_interlayer, wavelength=wl, kp=(0,0), depth=a*args.interlayer_depth, slicing_pow=2)
    Si = scattering_air_tmp(l.gx.shape, l, wl, args.interlayer_depth*a)

    Stot = multS(S, Si)

    if args.interlayer_shift:
        S2, _ = scattering_matrix(pw, l, "disc", [(x+args.interlayer_shift)*a, y*a, r*a], args.eps_holes, args.eps_layer, wavelength=wl, kp=(0,0), depth=a*args.layers_depth)
        Stot = multS(Stot, S2)
        pass
    else:
        Stot = multS(Stot, S)
        pass
    #Stot = S
    S_interf = scattering_interface(l, wl, kp=(0,0))
    Stot = multS(Stot, S_interf)
    j1plus, j1minus, j3plus = compute_currents(pin, Stot @ pin, l, wl)
    Ttot= np.abs(np.sum(j3plus) / np.sum(j1plus))
    transmission.append(Ttot)
np.save(f"{args.output}", transmission)
print("Done")