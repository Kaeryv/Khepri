import sys
sys.path.append(".")

import matplotlib.pyplot as plt
import numpy as np

import khepri as bt
N=256
pw = (7,1)
cl = bt.Crystal(pw, epsi=8, epse=8)
cl.add_layer_uniform("L0", 7.0, 0.9)
cl.set_device(["L0"], [False])

T = np.zeros(N)
R = np.zeros(N)
for i, f in enumerate(np.linspace(0.1, 0.8, N)):
    cl.set_source(1/f, 1, 1)
    cl.solve()

    R[i], T[i] = cl.poynting_flux_end()


plt.plot(R, label="R")
plt.plot(T, label="T")
plt.legend()
plt.show()

