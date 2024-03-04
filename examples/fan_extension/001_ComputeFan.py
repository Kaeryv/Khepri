'''
    This script computes S2_tilda and S1_tilda on G1 and G2 (rotated) subspaces resp.
    It is designed for array jobs, every result is a different file.
'''

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-angle", type=str, default="0:45:45", help="Angles to compute [degrees].")
parser.add_argument("-freq", type=str, default="0.7:0.84:80", help="Reduced frequencies to compute [c/a]")
parser.add_argument("-pw", nargs="+", type=int, required=True, help="Plane waves [x,y]")
parser.add_argument("-output", type=str, required=True, help="Output directory [empty folder]")
parser.add_argument("-range", type=int, nargs="+", required=True, help="Computation range: start, end")
parser.add_argument("-a", default=1e-7, type=float, help="Lattice constant [m]")
parser.add_argument("-hole_radius", default=0.25, type=float, help="Holes Radius [a]")
parser.add_argument("-hole_position", default=[0.5, 0.5], type=float, nargs="+", help="Holes Position x y [a]")
parser.add_argument("-layers_depth", default=0.2, type=float, help="Layers depth [a]")
parser.add_argument("-eps_emergent", default=1.0, type=float, help="Emergent Epsilon")
parser.add_argument("-eps_incident", default=1.0, type=float, help="Incident Epsilon")
parser.add_argument("-eps_layer", default=4.0, type=float, help="Layer Epsilon")
parser.add_argument("-eps_holes", default=1.0, type=float, help="Holes Epsilon")
args = parser.parse_args()

assert len(args.pw) == 2
assert args.a > 0.0

# We need the twisted and untwisted lattices
from bast.tmat.lattice import CartesianLattice
import numpy as np
import os

os.makedirs(f"ds/{args.output}/", exist_ok=True)
pw = args.pw
a = args.a
r = args.hole_radius
x, y = args.hole_position
disk_params = [ x*a, y*a, r*a] # Disk parameters [m]

eps_emergent = args.eps_emergent
eps_incident = args.eps_incident


from bast.tmat.scattering import scattering_matrix, scattering_interface
from bast.tools import c
from bast.tmat.matrices import multS

def get_smatrix(lattice, wavelength, kpoint):
    '''
    Compute a single S matrix at given k point and wavelength in lattice.
    '''
    S_interf = scattering_interface(lattice, wavelength, kpoint)
    S, _ = scattering_matrix(pw, lattice, "disc", disk_params, args.eps_holes, args.eps_layer, 
                             wavelength=wavelength, kp=kpoint, depth=a*args.layers_depth)

    # Pour introduire un angle, ajouter un kinc+kpoint ci-dessus
    # Se servir de lattice.kp_angle pour obtenir kinc
    return multS(S, S_interf)

def twisted_layers(pw, wl, alpha):
    '''
    Compute the little matrices in each basis (rotated and normal one).
    Tilda matrices.
    '''
    l1 = CartesianLattice(pw, (a, 0), (0., a), eps_incident, eps_emergent, dtype=np.float64)
    l2 = CartesianLattice(pw, (a, 0), (0., a), eps_incident, eps_emergent, dtype=np.float64)
    l2.rotate(alpha) # Rotate the basis l2 by alpha degrees
    
    S1s = [ get_smatrix(l1, wl, g2) for g2 in l2.g_vectors() ]
    S2s = [ get_smatrix(l2, wl, g1) for g1 in l1.g_vectors() ]

    return S1s, S2s

# Let's get all the alpha, wl pairs in a list.
import bast.misc as bm
from itertools import product
f = np.linspace(*bm.str2linspace_args(args.freq)) * c / a
wavelength = c / f
angles = np.linspace(*bm.str2linspace_args(args.angle))
wavelength_angle_pairs = np.asarray(list(product(wavelength, angles)))

# Compute the ones targetted in this job using the range given.
for i in range(*args.range):
    wl, ta = wavelength_angle_pairs[i, :]
    print(f"At iteration {i} over {len(wavelength)*len(angles)}." )
    S1s, S2s = twisted_layers(pw, wl, ta)
    np.savez_compressed(f"ds/{args.output}/Ss_{i:05d}.npz", S1s=S1s, S2s=S2s)

