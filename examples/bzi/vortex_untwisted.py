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

bzi = int(sys.argv[1])
bzs = (bzi,bzi)
def gen_bzi_grid(shape, a=1):
	si, sj = 1 / shape[0], 1 / shape[1]
	i, j = np.meshgrid(
	    np.arange(-0.5 + si / 2, 0.5, si),
	    np.arange(-0.5 + sj / 2, 0.5, sj),
	    )
	return np.stack([2*np.pi/a * i, 2*np.pi/a*j])

kbz = gen_bzi_grid(bzs)

m=bzs[0]
NS = 51
x = np.linspace(0*m, 1*m, NS*bzs[0])
y = np.linspace(0*m, 1*m, NS*bzs[1])
X, Y = np.meshgrid(x,y)
Z = np.zeros_like(X)
wl = 0.63# 1 / 0.518
theta = 0.0* np.pi

kp = compute_kplanar(1, wl, theta, 0)
kbz[0] += kp[0].real
kbz[1] += kp[1].real


#source_real=shifted_rotated_fields(_paraxial_gaussian_field_fn, X, Y, Z, wl, np.max(x)/2, np.max(y)/2, -3,theta,0.0*np.pi,0)
source_real = shifted_rotated_fields(_paraxial_laguerre_gaussian_field_fn, X, Y, Z, wl, np.max(x)/2, np.max(y)/2, -3, theta, 0, 0, l=1, p=0)

source_real = np.asarray(source_real)
source_real = np.swapaxes(source_real, 0, 2)
source_real = np.swapaxes(source_real, 1, 3)

pw = (3,3)
canvas_size = (512,512)

pattern = Drawing(canvas_size, 12)
pattern.disc((0,0), 0.4, 1)

e1 = Expansion(pw)
crystals = list()
kbz2 = kbz.reshape(2, -1).T
# Add a global kp to those if there is an incidence angle for the gaussian beam
for kp in kbz2:
    cl = Crystal(pw, epse=1)
    ss = 1
    cl.add_layer_uniform("S1", 1, 1/ss)
    cl.add_layer_uniform("S2", 1, 1/ss)
    #cl.add_layer_pixmap("Scyl", pattern.canvas(), 0.55/ss)
    #cl.set_stacking(["Scyl", "S1", "Scyl"])
    cl.set_stacking(["S1", "S2"])
    cl.set_source(wl, np.nan, np.nan, kp=kp)
    cl.solve()
    crystals.append(cl)


def amplitudes_from_fields(fields, kp, x, y):
    kxi, kyi = kp
    # Find mode coefficients
    phase = np.exp(1j*(kxi*x+kyi*y)) # There is no incidence angle.
    F = fields / phase[..., np.newaxis, np.newaxis]
    F = np.asarray(np.split(F, bzs[0], axis=1))
    F = np.asarray(np.split(F, bzs[1], axis=1))
    #F = np.swapaxes(F, 0,1)

    F = F.reshape(-1, NS, NS, 2, 3)
    F = F[..., :2]
    for i, f in enumerate(F):
        F[i] = np.fft.fftshift(np.fft.fft2(f, axes=(0,1))) / F.shape[0]/F.shape[1]
    F = np.sum(F, 0)
    return F
Fs = list()
for kp in kbz2:
     Fs.append(amplitudes_from_fields(source_real, kp, X, Y))
Fs=np.asarray(Fs)

#R0 = layer_eigenbasis_matrix(W0, V0)
pp = (pw[0]-1)//2
# Apply truncation
c = (NS -1) // 2
print("Fs0", Fs[0].shape)
Fs = [ F[c-pp:c+pp+1,c-pp:c+pp+1] for F in Fs ]

Fs = [ F.reshape(pw[0]**2, 2, 2) for F in Fs ]
Fs = [np.asarray([F[:, 0, 0],F[:, 0, 1],F[:, 1, 0],F[:, 1, 1]]).flatten() for F in Fs]
#F = np.linalg.solve(R0, F)
Fs = [ np.split(F, 2)[0] for F in Fs ]
x = np.linspace(0, bzs[0], 256)
y = np.linspace(0, bzs[1], 256)
x, y = np.meshgrid(x, y)

from tqdm import tqdm
from matplotlib.animation import FuncAnimation
print(x.shape, y.shape)
# Do BZI
zmax = cl.stack_positions[-1]
import os

fields = list()
for c, F, kp in zip(crystals, Fs, kbz2):
    E, H = c.fields_coords_xy(x, y, 1, F, use_lu=False)
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
fig.savefig("vortex_untwisted.png")