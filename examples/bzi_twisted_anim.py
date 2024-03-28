import sys
sys.path.append(".")
import numpy as np
from bast.crystal import Crystal

import numpy as np

import matplotlib.pyplot as plt
from bast.draw import Drawing
from bast.layer import Layer
from bast.expansion import Expansion
from bast.beams import _paraxial_gaussian_field_fn, shifted_rotated_fields, _paraxial_laguerre_gaussian_field_fn
from bast.tools import compute_kplanar
from bast.tools import rotation_matrix as rot
def size(angle):
    R = rot(np.pi/2)
    AM = 1/2 / np.sin(angle/2)* R @ np.eye(2)
    return np.linalg.norm(AM[0])

NS = 71
wl = 0.63
theta = 0.0*np.pi
twist_angle = 3
twcell = size(np.deg2rad(twist_angle))
override=sys.argv[4] == "override"
ar=4
zres = 256
xyres = 512


pw = (3,3)
# canvas_size = (31,31)

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

kbz = gen_bzi_grid(bzs, a=twcell)
kp = compute_kplanar(1, wl, theta, 0)
kbz[0] += kp[0].real
kbz[1] += kp[1].real

kbz2 = kbz.reshape(2, -1).T

m=bzs[0]
x = np.linspace(0*m, twcell*m, NS*bzs[0])
y = np.linspace(0*m, twcell*m, NS*bzs[1])
X, Y = np.meshgrid(x,y)
Z = np.zeros_like(X)
from functools import partial
field_fun = partial(_paraxial_gaussian_field_fn, beam_waist=10)
source_real=shifted_rotated_fields(field_fun, X, Y, Z, wl, np.max(x)/2, np.max(y)/2, -3,theta,0.0*np.pi,0)

source_real = np.asarray(source_real)
source_real = np.swapaxes(source_real, 0, 2)
source_real = np.swapaxes(source_real, 1, 3)

from bast.extension import ExtendedLayer
e1 = Expansion(pw)
e2 = Expansion(pw)
e2.rotate(twist_angle)
crystals = list()

from tqdm import tqdm
# Add a global kp to those if there is an incidence angle for the gaussian beam
for kp in tqdm(kbz2):
    twcl = Crystal.from_expansion(e1+e2, epse=4)
    etw = twcl.expansion
    twcl.add_layer_custom("Sref",     ExtendedLayer(etw, Layer.half_infinite(e1, "reflexion", 1),  e2.g_vectors, 1))
    twcl.add_layer_custom("up",  ExtendedLayer(etw, Layer.uniform(e1, 1.0,  3),              e2.g_vectors, 1))
    twcl.add_layer_custom("down", ExtendedLayer(etw, Layer.uniform(e2, 4.0, 3),              e1.g_vectors, 0))
    twcl.add_layer_custom("Strans",   ExtendedLayer(etw, Layer.half_infinite(e2, "transmission", 4), e1.g_vectors, 0))

    '''
        Define the device and solve.
    '''
    twcl.set_stacking(["up", "down"])
    twcl.set_source(wl, 1, 1, kp=kp)
    twcl.solve()
    crystals.append(twcl)

def plot_expansion_field(kx, ky, F):
    fig, ax = plt.subplots()
    size = np.abs(F)
    size /= np.max(size)
    ax.scatter(kx.real, ky.real, s=size*10)
    #for x,y,z in  zip(kx, ky, F):
    #    ax.text(x, y, f"{z.real:.1e},{z.imag:.1e}", fontsize=6)
    plt.show()
from bast.fields import dft
def amplitudes_from_fields(etw, fields, kp, x, y):
    kxi, kyi = kp
    # Find mode coefficients
    phase = np.exp(1j*(kxi*x+kyi*y)) # There is no incidence angle.
    F = fields / phase[..., np.newaxis, np.newaxis]
    F = np.asarray(np.split(F, bzs[0], axis=0))
    F = np.asarray(np.split(F, bzs[1], axis=2))
    F = np.swapaxes(F, 0, 1)

    F = F.reshape(-1, NS, NS, 2, 3)
    F = F[..., :2]
    Fnew = list()
    rx, ry, _ = etw.k_vectors((0,0), wl)
    k0 = 2*np.pi/wl

    for f in F:
        Fi = [[None, None], [None, None]]
        for j in range(f.shape[2]):
            for k in range(f.shape[3]):
                Fi[j][k] = dft(f[:,:, j, k], k0*rx, k0*ry, a=twcell) / f.shape[0]/ f.shape[1]
        Fnew.append(Fi)
    F = np.sum(Fnew, 0) # Sum over bzi pieces
    return F
Fs = list()
for kp, twcl in zip(kbz2, crystals):
     etw = twcl.expansion
     Fs.append(amplitudes_from_fields(etw,source_real, kp, X, Y))
Fs=np.asarray(Fs)

#R0 = layer_eigenbasis_matrix(W0, V0)
pp = (pw[0]-1)//2
# Apply truncation
print("Fs0", Fs[0].shape)
Fs = np.asarray([ np.moveaxis(F, 2, 0) for F in Fs])

Fs = [ F.reshape(len(etw.expansion_indices.T), 2, 2) for F in Fs ]
Fs = [np.asarray([F[:, 0, 0],F[:, 0, 1],F[:, 1, 0],F[:, 1, 1]]).flatten() for F in Fs]
#F = np.linalg.solve(R0, F)
Fs = [ np.split(F, 2)[0] for F in Fs ]
x = np.linspace(0, twcell*bzs[0], xyres)
y = np.ones_like(x)*0.5*bzs[0]*twcell
rx, ry, _ = etw.k_vectors(kp, wl)
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
print(x.shape, y.shape)
# Do BZI
zmax = twcl.stack_positions[-1]
zvals =np.linspace(0.0001, zmax, zres)
import os
fn = "tmpfields.npy"
if os.path.isfile(fn) and not override:
    fields = np.load(fn)
else:
    print("loading fields from disk")
    fields = list()
    for c, F, kp in zip(crystals, Fs, kbz2):
        E, H = c.fields_volume2(x, y, tqdm(zvals), incident_fields=F, use_lu=False)
        fields.append((E, H))
    fields = np.asarray(fields)
    np.save(fn, fields)
from bast.layer import Field

action = sys.argv[3]

shown = Field.X
if action == "static":
    fields = fields.mean(0)
    fields = fields[0, :, shown, :]
    print(np.max(fields.real), np.min(fields.real))
    fig, ax = plt.subplots(figsize=(5,6))
    image = ax.matshow(fields.real, cmap="RdBu",extent=[0, np.max(x), 0, zmax]) #, vmin=-1, vmax=1
    ax.set_aspect(ar)
    ax.set_xlabel("Length [µm]")
    ax.set_ylabel("Depth [µm]")
    fig.savefig("anim_twisted.png")

elif action == "explain":
    fig, axs = plt.subplots(*bzs)
    for i, (e, ax) in enumerate(zip(fields, axs.flatten())):
        ax.matshow(fields[i, 0, :, shown, :].real, cmap="RdBu")
        ax.axis("equal")
        ax.axis("off")
    fig.savefig("Bzi_explain.png")

elif action == "anim":
    fields = fields.mean(0)
    fields = fields[0, :, shown, :]
    K = np.exp(1j)
    fig, ax = plt.subplots(figsize=(6,6))
    image = ax.matshow(fields.real, cmap="RdBu",extent=[0, np.max(x), 0, zmax]) #, vmin=-1, vmax=1

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
    ax.set_xlim(np.max(x)/2-20,np.max(x)/2+20)
    def update(frame):
        image.set_data((fields * K**frame).real)
        return image,

    ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, int(sys.argv[2]), endpoint=False))
    ani.save("tanim.gif")