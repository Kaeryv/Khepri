import sys
sys.path.append(".")
import numpy as np
from bast.crystal import Crystal

import numpy as np
import numpy as np
from typing import Callable, Tuple

import matplotlib.pyplot as plt
from bast.draw import Drawing
from bast.layer import Layer
from bast.expansion import Expansion
from bast.beams import _paraxial_gaussian_field_fn, shifted_rotated_fields, _paraxial_laguerre_gaussian_field_fn
from bast.tools import compute_kplanar
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
NS = 31
x = np.linspace(0*m, 1*m, NS*bzs[0])
y = np.linspace(0*m, 1*m, NS*bzs[1])
X, Y = np.meshgrid(x,y)
Z = np.zeros_like(X)
wl = 0.63# 1 / 0.518
theta = 0.0* np.pi

kp = compute_kplanar(1, wl, theta, 0)
kbz[0] += kp[0].real
kbz[1] += kp[1].real


source_real=shifted_rotated_fields(_paraxial_gaussian_field_fn, X, Y, Z, wl, np.max(x)/2, np.max(y)/2, -3,theta,0.0*np.pi,0)
#source_real = shifted_rotated_fields(_paraxial_laguerre_gaussian_field_fn, X, Y, Z, wl, np.max(x)/2, np.max(y)/2, -3, theta, 0, 0, l=3, p=0)

source_real = np.asarray(source_real)
source_real = np.swapaxes(source_real, 0, 2)
source_real = np.swapaxes(source_real, 1, 3)

pw = (3,3)
canvas_size = (512,512)

pattern = Drawing(canvas_size, 12)
pattern.circle((0,0), 0.4, 1)

e1 = Expansion(pw)
crystals = list()
kbz2 = kbz.reshape(2, -1).T
# Add a global kp to those if there is an incidence angle for the gaussian beam
for kp in kbz2:
    cl = Crystal(pw, epse=1)
    ss = 2
    cl.add_layer_uniform("S1", 1, 2.1/ss)
    cl.add_layer_uniform("S2", 1, 1.1/ss)
    cl.add_layer_pixmap("Scyl", pattern.canvas(), 0.55/ss)
    cl.set_stacking(["Scyl", "S1", "Scyl"])
    stack = []
    stack.extend(["S1"]*ss)
    stack.extend(["S1"]*ss)
    #stack.extend(["Scyl"]*ss)
    #stack.extend(["S2"]*ss)
    #stack.extend(["Scyl"]*ss)
    stack.extend(["S2"]*ss)
    stack.extend(["S2"]*ss)
    cl.set_stacking(stack)
    cl.set_source(wl, np.nan, np.nan, kp=kp)
    cl.solve()
    crystals.append(cl)


def amplitudes_from_fields(fields, kp, x, y):
    kxi, kyi = kp
    # Find mode coefficients
    phase = np.exp(1j*(kxi*x+kyi*y)) # There is no incidence angle.
    F = fields / phase[..., np.newaxis, np.newaxis]
    F = np.asarray(np.split(F, bzs[0], axis=0))
    F = np.asarray(np.split(F, bzs[1], axis=2))
    F = np.swapaxes(F, 0,1)

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
x = np.linspace(0, 1*bzs[0], 512)
y = np.ones(512)*0.5*bzs[0]

from tqdm import tqdm
from matplotlib.animation import FuncAnimation
print(x.shape, y.shape)
# Do BZI
zmax = cl.stack_positions[-1]
zres = 256
zvals =np.linspace(0.0001, zmax, zres)
import os
fn = "tmpfields.npy"
if os.path.isfile(fn):
    fields = np.load(fn)
else:
    print("loading fields from disk")
    fields = list()
    for c, F, kp in zip(crystals, Fs, kbz2):
        E, H = c.fields_volume2(x, y, tqdm(zvals), incident_fields=F)
        fields.append((E, H))

    fig, axs = plt.subplots(*bzs)
    fields = np.asarray(fields)
    for i, (e, ax) in enumerate(zip(fields, axs.flatten())):
        ax.matshow(fields[i, 0, :, 0, :].real, cmap="RdBu")
        ax.axis("equal")
        ax.axis("off")
    fig.savefig("Bzi_explain2.png")
    fields = np.asarray(fields).mean(0)
    np.save(fn, fields)
from bast.layer import Field

shown = Field.X
if shown <= Field.Z:
    fields = fields[0, :, shown, :]
elif shown==Field.POYNTING:
    print("fields.shape", fields.shape)
    fields = np.swapaxes(fields, 3, 2)
    fshape = fields.shape
    #fields[:, :, 2, :] *= 10
    fields = fields.reshape(2, -1, 3)
    print("fields.shape", fields.shape)
    poynting = -np.cross(fields[0], np.conj(fields[1])).imag
    poynting = poynting.reshape(fshape[1], fshape[2], 3)
    print(poynting.shape)
    fig, ax = plt.subplots(figsize=(6,8))
    ssx = 16
    ssz = 8
    Pnorm = np.sqrt(np.sum(poynting[:,:,:]**2, axis=2)).real
    image = ax.matshow(Pnorm, cmap="hot",extent=[0, np.max(x), 0, zmax]) #, vmin=-1, vmax=1
    X, Z = np.meshgrid(x[::ssx], zvals[::ssz])
    print(X.shape, poynting[::ssz,::ssx,1].shape)
    #ax.streamplot(X, Z, poynting[::ssz,::ssx,0], poynting[::ssz,::ssx,2],
    #          color="gray", density=(0.6,1.5), broken_streamlines=False)
    Pnorm = np.sqrt(poynting[::ssz,::ssx,1]**2 +  poynting[::ssz,::ssx,2]**2).mean()
    ax.quiver(X, np.max(Z)-Z, poynting[::ssz,::ssx,0], poynting[::ssz,::ssx,2], pivot="mid",scale=50*Pnorm, width=0.005,color="k")
    
    fig.savefig("poynting.png")
    exit()


K = np.exp(1j)
fig, ax = plt.subplots(figsize=(6,6))
image = ax.matshow(fields.real, cmap="RdBu",extent=[0, np.max(x), 0, zmax]) #, vmin=-1, vmax=1
eps = pattern.canvas()
eps -= np.min(eps)
eps /= np.max(eps)
eps *= 0.55
eps = eps[:, eps.shape[0]//2]
eps = np.tile(eps, bzs[0])
#ax.plot(np.linspace(0,bzs[0], len(eps)), 4.2+eps, "k-")
#ax.plot(np.linspace(0,bzs[0], len(eps)), 4.2+0.55+1.1+eps, "k-")
# ax.axhline(4.2, color='k')
# ax.axhline(4.2+0.55, color='k')
# ax.axhline(4.2+0.55+1.1, color='k')
# ax.axhline(4.2+0.55+1.1+0.55, color='k')

ax.text(0.2, 3.5, "$\\varepsilon=4$")
ax.text(0.2, 5.5, "$\\varepsilon=1$")
ax.set_aspect(1)
ax.set_xlabel("Length [µm]")
ax.set_ylabel("Depth [µm]")
def update(frame):
    image.set_data((fields * K**frame).real)
    return image,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, int(sys.argv[2]), endpoint=False))
ani.save("anim.gif")