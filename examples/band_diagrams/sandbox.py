import sys
sys.path.append(".")
from bast.tools import  nanometers
from bast.tmat.matrices import matrix_s
from bast.tmat.lattice import CartesianLattice
import numpy as np
from bast.tmat.scattering import scattering_matrix

pw = (40, 40)
nx, ny = pw[0] // 2, pw[1] // 2
a  = nanometers(100)
lattice = CartesianLattice(pw, a1=(a, 0.0), a2=(0.0, a), eps_emerg=1.0, dtype=np.float32)
wavelength = nanometers(200)
theta = 0
phi = 0
import time

start =time.time()
S0, bf = scattering_matrix(pw, lattice, "rectangle", [0, 0, 0.5*a, 0.5*a], island_eps=3.0, eps_host=1.0, wavelength=wavelength, depth=0.5*a, slicing_pow=3, kp=(np.pi/2/a,np.pi/3/a))
stop = time.time()
print(S0[0,0])
print(S0.dtype)
print(stop - start, "seconds")

    
