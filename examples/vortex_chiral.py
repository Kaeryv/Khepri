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
import psutil
import os

process = psutil.Process(os.getpid())
fdepth = 5

# Worse as angle increases
twist_angle=float(sys.argv[3])


def size(angle_deg):
    angle = np.deg2rad(angle_deg)
    #AM = moire_lattice(angle_deg)
    #return np.linalg.norm(AM[0])
    return float(sys.argv[4])/((1/np.cos(angle))-1)
    #return float(sys.argv[4])/np.sqrt(1-np.cos(2*angle))

def moire_lattice(angle_deg):
    angle = np.deg2rad(angle_deg)
    #R = rot(np.pi/2)
    #R1 = rot(angle/2)
    #R2 = rot(-angle/2)
    #AM = 0.5 / np.sin(angle/2)* R @ np.eye(2)
    #A = R1 @ R1 @ np.eye(2)
    #AM = np.linalg.inv(R1-R2) @ np.eye(2)
    AM = rot(angle/2)@ np.eye(2)
    AM *= size(angle_deg)
    return AM

twisted = twist_angle != 0
cell_size =  size(twist_angle) if twisted else 1
print(cell_size)

bzi = int(sys.argv[1])
pwx = int(sys.argv[2])
pw = (pwx, pwx)

kbz = gen_bzi_grid((bzi, bzi), a=cell_size, reciproc=None).reshape(2, -1).T

e1 =Expansion(pw)
e2 =Expansion(pw)
e2.rotate(twist_angle)
e = e1+e2
gnorm =  np.linalg.norm(e.g_vectors, axis=0)
asrt = np.argsort(gnorm)
#for i in range(1,5):
#    print(gnorm[asrt[i]])
kmin = gnorm[asrt[2]]
cell_size = 4*np.pi / kmin
print(cell_size)
kbz = gen_bzi_grid((bzi, bzi), a=cell_size, reciproc=None).reshape(2, -1).T
t = np.linspace(0, np.pi*2)
kbz = (rot(np.deg2rad(0)) @ kbz.T).T
if False:
    fig, ax = plt.subplots()
    e.plot(ax=ax)
    plt.plot(kmin/2*np.cos(t), kmin/2 * np.sin(t), 'k-')
    ax.scatter(*kbz.T, c='b')
    plt.show()
#exit()

NS = 31 #int(cell_size * 3)
wl = 1 / 0.735
theta = 0

'''
    Build and solve the crystal
'''
from bast.beams import amplitudes_from_fields
from bast.alternative import incident

fields = list()
pattern = Drawing((128,128), 9)
pattern.circle((0,0), 0.45/2, 1)
e1, e2 = Expansion(pw), Expansion(pw)
e2.rotate(twist_angle)
etw = e1 + e2
for kp in tqdm(kbz):
    cl = Crystal.from_expansion(etw)
    ss = 2
    #cl.a = cell_size
    cl.add_layer("Sref",  EL(etw, Layer.half_infinite(e1, "reflexion", 1),    e2.g_vectors, 1))
    cl.add_layer("S1",    EL(etw, Layer.pixmap(e1, pattern.canvas(), 0.22/ss),    e2.g_vectors, 1))
    cl.add_layer("Si",    EL(etw, Layer.uniform(e1, 1.0,  0.3/ss),               e2.g_vectors, 1))
    cl.add_layer("Si2",   EL(etw, Layer.uniform(e2, 1.0,  1.5),               e1.g_vectors, 0))
    cl.add_layer("S2",    EL(etw, Layer.pixmap(e2, pattern.canvas(), 0.22/ss),    e1.g_vectors, 0))
    cl.add_layer("Strans",EL(etw, Layer.half_infinite(e2, "transmission", 1), e1.g_vectors, 0))
    cl.layers["Si2"].fields = True
    cl.layers["Sref"].fields = True
    cl.prepare_fields(fdepth-0.1, fdepth+0.1)
    stack = []
    stack.extend(["S1"]*ss)
    stack.extend(["Si"]*ss)
    stack.extend(["S2"]*ss)
    stack.extend(["Si2"]*4)
    cl.set_stacking(stack)
    cl.set_source(wl, np.nan, np.nan, kp=kp)
    cl.solve()
    expansion = etw

    print("Solved crystal, memory at ", process.memory_info().rss/1024**3, "GB")

    '''
        Sources computation
    '''
    x = np.linspace(0, cell_size*bzi, NS*bzi, endpoint=True)
    y = np.linspace(0, cell_size*bzi, NS*bzi, endpoint=True)
    X, Y = np.meshgrid(x,y, indexing="ij")
    Z = np.zeros_like(X)
    
    zmax = cl.stack_positions[-2]
    print(f"Computing sources {zmax=}")
    source_real=shifted_rotated_fields(_paraxial_gaussian_field_fn, X, Y, Z, wl, np.max(x)/2, np.max(y)/2, 1, theta,0.0*np.pi, 0.5*np.pi/2, beam_waist=10)
    #source_real = shifted_rotated_fields(_paraxial_laguerre_gaussian_field_fn, X, Y, Z, wl, np.max(x)/2, np.max(y)/2, -1, theta, 0, np.pi/4, l=1, p=0, w0=0.7*cell_size)
    
    source_real = np.asarray(source_real)
    source_real = np.swapaxes(source_real, 0, 2)
    source_real = np.swapaxes(source_real, 1, 3)
    
    F = amplitudes_from_fields(source_real, expansion, wl, kp, X, Y, (bzi, bzi), a=cell_size)


    F = np.split(F.flatten(), 2)[0] 
    x = np.linspace(2.5*cell_size, cell_size*(bzi-2.5), 256)
    y = np.linspace(2.5*cell_size, cell_size*(bzi-2.5), 256)
    #y = np.linspace(30, 60, 64)
    print(x[0], x[-1], cell_size*bzi)
    x, y = np.meshgrid(x, y, indexing="ij")
    
    print("Computing fields")

    E, H = cl.fields_coords_xy(x, y, fdepth, F, use_lu=False)
    fields.append((E, H))

fields = np.asarray(fields)
np.savez_compressed("tw_fields.npz", fields)

if False:
    fig, axs = plt.subplots(bzi, bzi)
    fig2, axs2 = plt.subplots(bzi, bzi)
    for field, ax1, ax2, kp in zip(fields, axs.flat,axs2.flat, kbz):
        f = field[0, 0, :, :]
        nf = np.abs(f)
        nf /= np.max(nf)
        ax1.matshow(np.angle(f), cmap="hsv", vmin=-np.pi, vmax=np.pi)
        ax2.matshow(np.abs(f), cmap="hot")
        ax2.axis("off")
        ax1.axis("off")
    fig2.savefig("explode_bzi.png")
    fig.savefig("explode_bzi_angle.png")
    plt.show()
fields = fields.sum(0)
fields = fields[0, 0, :, :]
sr = source_real[..., 0, 0]

fig, axs = plt.subplots(2, 2, figsize=(5,5))
normfields = np.abs(fields)
normfields /= np.max(normfields)
#np.abs
image = axs[0,0].matshow(np.abs(fields), cmap="hot",extent=[np.min(x), np.max(x), np.min(x), np.max(y)]) #, vmin=-1, vmax=1
plt.colorbar(image)
image = axs[0,1].matshow(np.angle(fields), cmap="hsv",extent=[0, np.max(x), 0, np.max(y)], alpha=normfields, vmin=-np.pi, vmax=np.pi) #, vmin=-1, vmax=1
normsr = np.abs(sr)
normsr /= np.max(normsr)
image = axs[1,0].matshow(normsr, cmap="hot",extent=[0, np.max(x), 0, np.max(y)]) #, vmin=-1, vmax=1
image = axs[1,1].matshow(np.angle(sr), cmap="hsv",extent=[0, np.max(x), 0, np.max(y)], alpha=normsr) #, vmin=-1, vmax=1
#ax1.set_aspect(ar)
axs[0,0].set_xlabel("Length [µm]")
axs[0,0].set_ylabel("Width [µm]")
fig.savefig("vortex_chiral.png")
fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(normfields, cmap="hot",extent=[0, np.max(x), 0, np.max(y)]) #, vmin=-1, vmax=1
ax.axis("off")
plt.savefig("hres.png")
