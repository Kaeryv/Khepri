import sys
sys.path.append(".")

import matplotlib.pyplot as plt
import numpy as np
from bast.alternative import incident, scattering_structured_layer, scattering_uniform_layer, redheffer_product, free_space_eigenmodes
from bast.expansion import Expansion
from bast.tools import c
from math import pi
from cmath import sqrt as csqrt
from numpy.linalg import inv
from tqdm import tqdm


er = 12
r  = 0.4
d  = 0.55
di = 1.35

N = 256
eps = np.ones((N, N))*er
x = np.linspace(-0.5, 0.5, N)
y = np.linspace(-0.5, 0.5, N)
X, Y = np.meshgrid(x, y)
eps[np.sqrt(X**2+Y**2)< r] = 1.0


freqsn = np.linspace(0.49, 0.59, 200)
kp = (0,0)
TTTs = list()
for di in [1.35,1.1,0.95,0.85,0.75,0.65,0.55]:
    TTs = list()
    for f in tqdm(freqsn):
        #fn = 0.8
        wavelength = 1 / f
        pw = (9,9)
        e = Expansion(pw)
        S = scattering_structured_layer(e, kp, wavelength, eps, d)
        Si = scattering_uniform_layer(e, kp, wavelength, 1.0, di)
        Stot = redheffer_product(S, redheffer_product(Si, S))
        epsi=1
        kp = (0,0)
        k0 = 2*pi/wavelength
        kzi = np.conj(csqrt(k0**2*epsi-kp[0]**2-kp[1]**2))
        esrc = incident(pw, 1, 0, (kp[0], kp[1], kzi))
        S21 = Stot[1,0]
        kx, ky, kz = e.k_vectors(kp, wavelength)
        W0, VO = free_space_eigenmodes(kx, ky)
        etm  = W0 @  S21 @ inv(W0) @ esrc
        tx, ty = np.split(etm, 2)
        tz = -  (kx*tx+ky*ty) / kz
        t = kz.real @ (np.abs(tx)**2+np.abs(ty)**2+np.abs(tz)**2)
        TT = np.sum(t)
        TTs.append(TT) 
    TTTs.append(TTs)


cmap = iter(plt.cm.tab20(np.arange(10)))
colors = iter(["darkred", "orangered", "orange", "gold", "limegreen", "forestgreen", "cadetblue", "royalblue"])

for TT, hi in zip(TTTs, [1.35,1.1,0.95,0.85,0.75,0.65,0.55]):
    plt.plot(freqsn, TT, c=next(cmap), label=f"h={hi}")

plt.xlabel("Frequency [c/a]")
plt.ylabel("Transmission")
plt.legend(loc="lower left")
plt.savefig("Spectrum.svg")