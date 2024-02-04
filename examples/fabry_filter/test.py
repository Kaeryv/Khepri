import sys
sys.path.append(".")
import numpy as np
import matplotlib.pyplot as plt
from bast.lattice import CartesianLattice
from bast.scattering import scattering_matrix, scattering_interface
from bast.tools import c, incident, compute_fluxes


d = np.loadtxt("examples/fabry_filter/fabry.txt")
#exit()
pw = (1,1)
a = 1e-7
lattice = CartesianLattice(pw, (a,0), (0,a), 1.0, 1.4**2)
wavelengths = np.linspace(300, 730, 200)
refl = list()
for wl in wavelengths:
    S1 = scattering_matrix(pw, lattice, "uniform", [], 1.35**2, 1.35**2, wl*1e-9, (0,0), depth=70e-9, slicing_pow=1)
    #S3 = scattering_matrix(pw, lattice, "uniform", [], 1.88**2, 1.88**2, wl*1e-9, (0,0), depth=20e-9, slicing_pow=2)
    S2 = scattering_matrix(pw, lattice, "uniform", [], 2.4**2, 2.4**2, wl*1e-9, (0,0), depth=120e-9, slicing_pow=1)
    
    S = S1 @ S2 @ S1 @ S2 @ S1 @ S2 @ S1

    Si = scattering_interface(lattice, wl, kp=(0,0))
    S = S @ Si


    pin = incident(pw, p_pol=1, s_pol=0)
    pout = S @ pin
    R,T,A =  compute_fluxes(lattice, wl, pin, pout)
    refl.append(T)

plt.plot(wavelengths, refl)
plt.plot(d.T[0], d.T[1]/7000)
plt.gca().set_ylim(0,1)
plt.gca().set_xlim(400,730)
plt.show()

#1.3 * l1 + 2.4 * l2 = 560 / 2 = 280
#1.3 * 
# 130 + 240 = 370 ?= 280
# 520 - 430 = 90nm de cryo 100nm de Zns
# 310-420 = 
