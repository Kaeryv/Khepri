import sys
sys.path.append(".")
from bast.alternative import incident
import numpy as np
import matplotlib.pyplot as plt
from bast.beams import _paraxial_gaussian_field_fn, shifted_rotated_fields, _paraxial_laguerre_gaussian_field_fn

N=64
x = y = np.linspace(0, 3, N)
z = np.linspace(0, 6, N*4)
from itertools import product
X, Y, Z = (e.flatten() for e in np.meshgrid(x, y, z))
wl = 1/0.7
theta = 0
#Z = np.zeros_like(X)
#fields = shifted_rotated_fields(_paraxial_laguerre_gaussian_field_fn, X, Y, Z, 0.6, 0.,0.0, 0, 0, 0, 0, l=4, p=1)
source_real=shifted_rotated_fields(_paraxial_gaussian_field_fn, X, Y, Z, wl, np.max(x)/2, np.max(y)/2, 3, theta,0.0*np.pi, 0, beam_waist=1.5)
fields = np.reshape(source_real, (2, 3, N, N, N*4))
#fields = np.asarray(fields).reshape((len(x), len(y), 2, 3))
fig, (ax1, ax2) = plt.subplots(2)
ax1.matshow(np.angle(fields[0,0, :, N//2, :]), cmap="hsv")
ax2.matshow(np.abs(fields[0,0, :, N//2, :]), cmap="hot")
plt.savefig("VizSource.png")
#plt.show()