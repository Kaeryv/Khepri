import sys
sys.path.append(".")

from bast.crystal import Crystal
from bast.draw import Drawing
from bast.expansion import Expansion
from bast.layer import Layer
import matplotlib.pyplot as plt
import numpy as np
from bast.crystal import Crystal
from tqdm import tqdm
from bast.misc import coords
from bast.tools import rotation_matrix as rot
from glob import glob

'''
    Define the pattern (common to both layers)
'''
pattern = Drawing((256,256), 4)
pattern.circle((0,0), 0.25, 1.0)

'''
    Define the crystal layers. 
    Be careful that layers with different twist angles are different objects.
'''
def solve_crystal(wl, twist_angle, polar=(1,1), theta=0, phi=0, pw=(3,3), fields=False):
    e1 = Expansion(pw)
    e2 = Expansion(pw)
    e2.rotate(twist_angle)

    twcl = Crystal.from_expansion(e1+e2)
    etw = twcl.expansion
    twcl.add_layer("Sref",   Layer.half_infinite(e1, "reflexion", 1), True)
    twcl.add_layer("Scup", Layer.pixmap(e1, pattern.canvas(), 0.2), True)
    twcl.add_layer("Sair",   Layer.uniform(e1, 1.0, 0.3), True)
    twcl.add_layer("Scdo", Layer.pixmap(e2, pattern.canvas(), 0.2), True)
    twcl.add_layer("Strans", Layer.half_infinite(e2, "transmission", 1), True)

    if fields:
        twcl.set_device(["Scup", "Sair", "Scdo"], [True]*3)
    else:
        twcl.set_device(["Scup", "Sair", "Scdo"])
    twcl.set_source(wl, polar[0], polar[1], theta, phi)
    twcl.solve()
    return twcl

if sys.argv[1] == "RT":
    M = 60
    RT = np.zeros((M,2))
    angle = float(sys.argv[3])
    for i, f in enumerate(np.linspace(0.7, 0.83, M)):
        twcl = solve_crystal(1/f, angle)
        RT[i] = twcl.poynting_flux_end()

    fig, ax = plt.subplots()
    ax.plot(RT[:, 0], label="R")
    ax.plot(RT[:, 1], label="T")
    fig.savefig(sys.argv[2])
    if len(sys.argv) >= 5:
        np.save(sys.argv[4], RT)

if sys.argv[1] == "map":
    pattern = sys.argv[2]
    files = glob(pattern)
    num_angles = len(files)
    rtmap = list()
    for i in range(num_angles):
        rtmap.append(np.load(pattern.replace("*", str(i))))
    rtmap = np.asarray(rtmap).reshape(num_angles, -1, 2)
    fig, ax = plt.subplots()
    ax.matshow(rtmap[..., 0].T, origin="lower", vmin=0, vmax=1)
    fig.savefig(sys.argv[3])


if sys.argv[1] == "lg_fields":
    '''
        Get those fields in longitudinal plane.
    '''
    wl = 1 / 0.7
    ta = 6
    twcl = solve_crystal(wl, ta, polar=(1,1), pw=(5,5), fields=True)

    x, y, z = coords(0, 12, 0.5, 0.5, 1e-7, 1.5, (256, 1, 64))
    E, H = twcl.fields_volume(x, y, tqdm(z)) # Progress bar on z axis
    E = np.squeeze(E)

    '''
        Plot them.
    '''
    Exxz = E[:, 0, :].real
    Eyxz = E[:, 1, :].real
    Edisp = Exxz
    vmax = np.max(np.abs(Edisp))
    fig, ax = plt.subplots(figsize=(10,5))
    ax.matshow(Edisp, cmap="RdBu", origin="lower", vmin=-vmax, vmax=vmax, aspect=3, extent=[ 0, np.max(x), 0, np.max(z)])


    '''
        Overlay the structure (WIP).
    '''
    for z in [0.0, 0.4, 0.6, 0.9, 1.1, 1.5]:
        ax.axhline(z, color="k")
    eps = pattern.canvas()
    eps = eps[eps.shape[0]//2, :]
    eps -= 1
    eps /= eps.max()
    eps = np.tile(eps, 12)
    x = np.linspace(0, 12, len(eps))
    plt.fill_between(x, 0.4, eps*0.2+0.4, alpha=0.2, color="k")
    plt.fill_between(x, 0.4, eps*0.2+0.4+0.2+0.3, alpha=0.2, color="k")
    plt.xlabel("x[µm]")
    plt.ylabel("z[µm]")
    plt.gca().set_position([0.05, 0.05, 0.95, 0.95])
    plt.savefig(sys.argv[2])


if False:
    '''
        Get those fields in transversal plane
    '''
    #etw.plot()
    zmax = twcl.stack_positions[-2]
    depth = 0.35#zmax+0.9
    w = 6
    x, y, z = coords(-w, w, -w, w, depth, depth, (512, 512, 1))
    E, H = twcl.fields_volume2(x, y, z)
    E = np.squeeze(E)
    print(E.shape)
    Exxy = E[0]
    Eyxy = E[1]
    Ezxy = E[2]
    Edisp =  Exxy#**2+Eyxy**2#+Ezxy**2
    vmax = np.max(np.abs(Edisp.real))
    fig, (ax1,ax2) = plt.subplots(2, figsize=(10,10))
    im = ax1.matshow(Edisp.real, cmap="bwr", origin="lower", vmin=-vmax, vmax=vmax, extent=[ np.min(x), np.max(x), np.min(y), np.max(y)])
    #im = ax1.matshow(np.abs(Edisp), cmap="inferno", origin="lower", vmin=0, vmax=vmax, extent=[ np.min(x), np.max(x), np.min(y), np.max(y)])
    alpha = np.abs(Edisp) / np.max(np.abs(Edisp))
    im2 = ax2.matshow(np.angle(Edisp), cmap="hsv", origin="lower", vmin=-np.pi, vmax=np.pi, extent=[ np.min(x), np.max(x), np.min(y), np.max(y)], alpha=alpha)
    plt.colorbar(im)
    plt.colorbar(im2)
    ax1.axis("square")
    ax2.axis("square")
    plt.savefig("twisted_fields_transversal.png")
