import sys
sys.path.append(".")
import numpy as np
from bast.crystal import Crystal
import matplotlib.pyplot as plt
from bast.draw import Drawing
from bast.layer import Layer
from bast.expansion import Expansion
from bast.extension import ExtendedLayer as EL
from bast.beams import _paraxial_gaussian_field_fn, shifted_rotated_fields, _paraxial_laguerre_gaussian_field_fn
from bast.tools import compute_kplanar
import sys
from tqdm import tqdm
from bast.beams import gen_bzi_grid
from bast.tools import rotation_matrix as rot
from bast.tools import reciproc

# Worse as angle increases
twist_angle=np.deg2rad(20)


def moire_lattice(angle):
    R = rot(np.pi/2)
    R1 = rot(angle/2)
    R2 = rot(-angle/2)
    #AM = 0.5 / np.sin(angle/2)* R @ np.eye(2)
    #A = R1 @ R1 @ np.eye(2)
    AM = np.linalg.inv(R1-R2) @ np.eye(2)
    return AM

def size(angle_deg):
    #AM = moire_lattice(angle_deg)
    #return np.linalg.norm(AM[0])
    return 1/((1/np.cos(np.deg2rad(angle_deg)))-1)
twisted = twist_angle != 0
#cell_size = 15.38 + 0*size(twist_angle) if twisted else 1
cell_size = size(twist_angle) if twisted else 1
print(cell_size)
#cell_size = 15.3
#print(cell_size)
#exit()

if twisted:
    moire_basis = moire_lattice(twist_angle)
    moire_reciproc = reciproc(moire_basis[0], moire_basis[1]) 
else:
    moire_reciproc=None

bzi = int(sys.argv[1])

kbz = gen_bzi_grid((bzi, bzi), a=cell_size, reciproc=None).reshape(2, -1).T

NS = int(cell_size * 11)
wl = 0.6
theta = 0.0
x = np.linspace(0, cell_size*bzi, NS*bzi, endpoint=True)
y = np.linspace(0, cell_size*bzi, NS*bzi, endpoint=True)
X, Y = np.meshgrid(x,y, indexing="ij")
Z = np.zeros_like(X)

#kp = compute_kplanar(1, wl, theta, 0)
#kbz[:, 0] += kp[0].real
#kbz[:, 1] += kp[1].real


#source_real=shifted_rotated_fields(_paraxial_gaussian_field_fn, X, Y, Z, wl, np.max(x)/2, np.max(y)/2, -3,theta,0.0*np.pi,0)
source_real = shifted_rotated_fields(_paraxial_laguerre_gaussian_field_fn, X, Y, Z, wl, np.max(x)/2, np.max(y)/2, -2, theta, 0, 0, l=1, p=0, w0=0.7*cell_size)

source_real = np.asarray(source_real)
source_real = np.swapaxes(source_real, 0, 2)
source_real = np.swapaxes(source_real, 1, 3)

pattern = Drawing((128,128), 8)
pattern.disc((0,0), 0.4, 1)
pwx = int(sys.argv[2])
pw = (pwx, pwx)
def get_crystals(kbz, twisted=False):
    crystals = list()
    if not twisted:
        e = Expansion(pw)
        for kp in kbz:
            cl = Crystal(pw, epse=1)
            ss = 1
            cl.add_layer_uniform("S1", 1, 1/ss)
            cl.add_layer_uniform("S2", 1, 1/ss)
            cl.set_stacking(["S1", "S2"])
            cl.set_source(wl, np.nan, np.nan, kp=kp)
            cl.solve()
            crystals.append(cl)
    else:
        e1, e2 = Expansion(pw), Expansion(pw)
        e2.rotate(twist_angle)
        etw = e1 + e2
        for kp in tqdm(kbz):
            cl = Crystal.from_expansion(etw)
            cl.a = cell_size
            cl.add_layer("Sref",  EL(etw, Layer.half_infinite(e1, "reflexion", 1), e2.g_vectors, 1))
            cl.add_layer("S1",    EL(etw, Layer.uniform(e1, 1.0,  0.5), e2.g_vectors, 1))
            cl.add_layer("S2",    EL(etw, Layer.uniform(e2, 1.0,  0.5), e1.g_vectors, 0))
            cl.add_layer("Strans",EL(etw, Layer.half_infinite(e2, "transmission", 1),e1.g_vectors, 0))
            cl.set_stacking(["S1", "S2"])
            cl.set_source(wl, np.nan, np.nan, kp=kp)
            cl.solve()
            crystals.append(cl)
        e = etw
    return crystals, e

crystals, expansion = get_crystals(kbz, twisted=twisted)

from bast.beams import amplitudes_from_fields
Fs = list()
print("Computing sources")
for kp in tqdm(kbz):
     Fs.append(amplitudes_from_fields(source_real, expansion, wl, kp, X, Y, (bzi, bzi), a=cell_size))
Fs=np.asarray(Fs)

Fs = [ np.split(F.flatten(), 2)[0] for F in Fs ]
#Fs = [F.flatten() for F in Fs]
'''
    Compute the fields after the structure.
'''
x = np.linspace(0, cell_size*bzi, 200)
y = np.linspace(0, cell_size*bzi, 200)
x, y = np.meshgrid(x, y, indexing="ij")

print("Computing fields")
fields = list()
for c, F, kp in zip(tqdm(crystals), Fs, kbz):
    E, H = c.fields_coords_xy(x, y, 0.1, F, use_lu=False)
    fields.append((E, H))
fields = np.asarray(fields)


fig, axs = plt.subplots(bzi, bzi)
fig2, axs2 = plt.subplots(bzi, bzi)
for field, ax1, ax2, kp in zip(fields, axs.flat,axs2.flat, kbz):
    f = field[0, 0, :, :]
    nf = np.abs(f)
    nf /= np.max(nf)
    ax1.matshow(np.angle(f), cmap="hsv", vmin=-np.pi, vmax=np.pi)
    ax2.matshow(np.abs(f), cmap="hot")
fig.savefig("explode_bzi.png")
fig2.savefig("explode_bzi_angle.png")

fields = fields.mean(0)
fields = fields[0, 0, :, :]
sr = source_real[..., 0, 0]
fig, axs = plt.subplots(2, 2, figsize=(5,5))
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
