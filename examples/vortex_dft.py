import sys
sys.path.append(".")
import numpy as np
from khepri.crystal import Crystal

import numpy as np
import numpy as np
from typing import Callable, Tuple

import matplotlib.pyplot as plt
from khepri.draw import Drawing
from khepri.layer import Layer
from khepri.expansion import Expansion
from khepri.beams import _paraxial_gaussian_field_fn, shifted_rotated_fields, _paraxial_laguerre_gaussian_field_fn
from khepri.tools import compute_kplanar
import sys

from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from khepri.beams import gen_bzi_grid

bzi = int(sys.argv[1])
bzs = (bzi,bzi)

kbz = gen_bzi_grid(bzs).reshape(2, -1).T

m=bzs[0]
NS = 31
x = np.linspace(0*m, 1*m, NS*bzs[0], endpoint=False)
y = np.linspace(0*m, 1*m, NS*bzs[1], endpoint=False)
X, Y = np.meshgrid(x,y)
Z = np.zeros_like(X)
wl = 0.63# 1 / 0.518
theta = 0.0* np.pi

kp = compute_kplanar(1, wl, theta, 0)
kbz[:, 0] += kp[0].real
kbz[:, 1] += kp[1].real


#source_real=shifted_rotated_fields(_paraxial_gaussian_field_fn, X, Y, Z, wl, np.max(x)/2, np.max(y)/2, -3,theta,0.0*np.pi,0)
source_real = shifted_rotated_fields(_paraxial_laguerre_gaussian_field_fn, X, Y, Z, wl, np.max(x)/2, np.max(y)/2, -3, theta, 0, 0, l=2, p=0)

source_real = np.asarray(source_real)
source_real = np.swapaxes(source_real, 0, 2)
source_real = np.swapaxes(source_real, 1, 3)

pw = (3,3)
canvas_size = (512,512)

# Add a global kp to those if there is an incidence angle for the gaussian beam
def get_crystals_untwisted(kbz, twisted=False):
    e1 = Expansion(pw)
    crystals = list()
    for kp in kbz:
        cl = Crystal(pw, epse=1)
        ss = 1
        cl.add_layer_uniform("S1", 1, 1/ss)
        cl.add_layer_uniform("S2", 1, 1/ss)
        cl.set_stacking(["S1", "S2"])
        cl.set_source(wl, np.nan, np.nan, kp=kp)
        cl.solve()
        crystals.append(cl)
    return crystals, e1

crystals, e = get_crystals_untwisted(kbz, twisted=False)

from khepri.beams import amplitudes_from_fields
Fs = list()
for kp in kbz:
     Fs.append(amplitudes_from_fields(source_real, e, wl, kp, X, Y, bzs))
Fs=np.asarray(Fs)

Fs = [ np.split(F.flatten(), 2)[0] for F in Fs ]

'''
    Compute the fields after the structure.
'''
x = np.linspace(0, bzs[0], 256)
y = np.linspace(0, bzs[1], 256)
x, y = np.meshgrid(x, y)


fields = list()
for c, F, kp in zip(crystals, Fs, kbz):
    E, H = c.fields_coords_xy(x, y, 0.01, F, use_lu=False)
    fields.append((E, H))
fields = np.asarray(fields)

fields = fields.mean(0)
fields = fields[0, 0, :, :]
sr = source_real[..., 0, 0]
fig, axs = plt.subplots(2, 2, figsize=(5,6))
normfields = np.abs(fields)
normfields /= np.max(normfields)
image = axs[0,0].matshow(normfields, cmap="hot",extent=[0, np.max(x), 0, np.max(y)]) #, vmin=-1, vmax=1
image = axs[0,1].matshow(np.angle(fields), cmap="hsv",extent=[0, np.max(x), 0, np.max(y)], alpha=normfields) #, vmin=-1, vmax=1
normsr = np.abs(sr)
normsr /= np.max(normsr)
image = axs[1,0].matshow(normsr, cmap="hot",extent=[0, np.max(x), 0, np.max(y)]) #, vmin=-1, vmax=1
image = axs[1,1].matshow(np.angle(sr), cmap="hsv",extent=[0, np.max(x), 0, np.max(y)], alpha=normsr) #, vmin=-1, vmax=1
#ax1.set_aspect(ar)
axs[0,0].set_xlabel("Length [µm]")
axs[0,0].set_ylabel("Width [µm]")
fig.savefig("vortex_twisted.png")