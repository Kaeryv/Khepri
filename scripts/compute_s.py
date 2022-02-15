from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-h5", type=str, required=True)
parser.add_argument("-wl", type=float, required=True)
parser.add_argument("-a", type=float, default=100)
args = parser.parse_args()

from bast.tools import incident, compute_currents, nanometers
from bast.matrices import matrix_s, multS
from bast.lattice import CartesianLattice
import numpy as np
from math import prod
from bast.scattering import scattering_matrix,scattering_matrix_npy
from tqdm import tqdm
import os
import h5py

hf_in = h5py.File(args.h5, 'r')
fft_t = hf_in.get("struct/fft/twisted")[:]
fft_n = hf_in["struct/fft/not_twisted"][:]
multiplicity = hf_in["struct"].attrs["multiplicity"]
pw = hf_in["struct"].attrs["pw"]
hf_in.close()
folder = "./"
os.makedirs(folder, exist_ok=True)
nx, ny = pw[0] // 2, pw[1] // 2
a  = nanometers(args.a) * multiplicity
lattice = CartesianLattice(pw, a1=(a, 0.0), a2=(0.0, a), eps_emerg=1.0)
wavelength = nanometers(args.wl)
theta = 0
phi = 0

U = lattice.U(wavelength, theta_deg=theta, phi_deg=phi)
Ve = lattice.Ve(wavelength, theta_deg=theta, phi_deg=phi)
T_interface = U @ Ve
S_interface = matrix_s(T_interface)

S0, bf = scattering_matrix(pw, lattice, "rectangle", [0, 0, a, a], island_eps=1.0, eps_host=1.0, wavelength=wavelength, depth=0.5*a, slicing_pow=5, theta_deg=theta, phi_deg=phi)
S1, bf = scattering_matrix_npy(pw, lattice, fft_t, island_eps=1.0, eps_host=10.0, wavelength=wavelength, depth=0.5*a, slicing_pow=5, theta_deg=theta, phi_deg=phi)
S2, bf = scattering_matrix_npy(pw, lattice, fft_n, island_eps=1.0, eps_host=10.0, wavelength=wavelength, depth=0.5*a, slicing_pow=5, theta_deg=theta, phi_deg=phi)
S = multS(S1, S0)
S = multS(S, S2)

hf = h5.File("./out.hdf5", "w")
hf['S0'] = S0
hf['S1'] = S1
hf['S2'] = S2
hf['S_interface'] = S_interface
hf['S'] = S
hf.close()
#S_tot = multS(S, S_interface)

    
