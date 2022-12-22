import numpy as np
import matplotlib.pyplot as plt
from bast.lattice import CartesianLattice
from bast.tools import incident
pw = (3,3)
a = 2 * np.pi
l1 = CartesianLattice(pw, (a, 0), (0, a))
l2 = CartesianLattice(pw, (a, 0), (0, a))
l2.rotate(34)
l = l1 + l2

pin = incident(l.pw, 1.0/np.sqrt(2), p_pol=1j, s_pol=1)
for i, (gx, gy) in enumerate(l.g_vectors()):
    print(i, round(gx,2), round(gy,2), pin[i])