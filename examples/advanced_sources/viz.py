import sys
sys.path.append(".")
from bast.alternative import incident
import numpy as np
import matplotlib.pyplot as plt
from bast.beams import _paraxial_gaussian_field_fn, shifted_rotated_fields, _paraxial_laguerre_gaussian_field_fn

N=64
x = np.linspace(-1, 1, N)
y= x.copy()
from itertools import product
X, Y = (e.flatten() for e in np.meshgrid(x, y))
Z = np.zeros_like(X)
fields = shifted_rotated_fields(_paraxial_laguerre_gaussian_field_fn, X, Y, Z, 0.6, 0.,0.0, 0, 0, 0, 0, l=4, p=1)
fields = np.asarray(fields)
#fields = np.asarray(fields).reshape((len(x), len(y), 2, 3))
fig, (ax1, ax2) = plt.subplots(2)
ax1.matshow(np.angle(fields[0,0].reshape(N, N)), cmap="hsv")
ax2.matshow(np.abs(fields[0,0].reshape(N, N)), cmap="hot")
plt.savefig("VizSource.png")
#plt.show()