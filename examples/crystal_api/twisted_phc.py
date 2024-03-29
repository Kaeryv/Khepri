import sys
sys.path.append(".")

from bast.crystal import Crystal
from bast.extension import ExtendedLayer
from bast.draw import Drawing
from bast.expansion import Expansion
from bast.layer import Layer
import matplotlib.pyplot as plt
import numpy as np
from bast.crystal import Crystal
from tqdm import tqdm
from bast.misc import coords
from bast.tools import rotation_matrix as rot

'''
    Parameters
'''
pw = (7,7)
N = 256
frequency = 0.7195 # c/a
angle_deg = 8.1    # deg
polarization = (1, 1j) # norm
theta = 0
phi = 0

'''
    Define the pattern (common to both layers)
'''
canvas_size = (N,N)
pattern = Drawing(canvas_size, 4)
pattern.circle((0,0), 0.25, 1.0)

'''
    Define the twist.
'''
e1 = Expansion(pw)
e2 = Expansion(pw)
e2.rotate(angle_deg)
# e1.rotate(-angle_deg/2)

'''
    Define the crystal layers. (Be careful that layers with different twist angles are different objects.)
'''
twcl = Crystal.from_expansion(e1+e2)
etw = twcl.expansion
#twcl.expansion._gvectors = rot(np.deg2rad(-90/2)) @ etw.g_vectors
twcl.add_layer("Sref",     ExtendedLayer(etw, Layer.half_infinite(e1, "reflexion", 1),  e2.g_vectors, 1))
#twcl.add_layer("Sbuffer",  ExtendedLayer(etw, Layer.uniform(e1, 1.0, 0.4),              e2.g_vectors, 1))
twcl.add_layer("Scylup",   ExtendedLayer(etw, Layer.pixmap(e1, pattern.canvas(), 0.2),  e2.g_vectors, 1))
twcl.add_layer("Sair",     ExtendedLayer(etw, Layer.uniform(e1, 1.0, 0.3),              e2.g_vectors, 1))
twcl.add_layer("Scyldo",   ExtendedLayer(etw, Layer.pixmap(e2, pattern.canvas(), 0.2),  e1.g_vectors, 0))
#twcl.add_layer("Sbuffer2", ExtendedLayer(etw, Layer.uniform(e2, 1.0, 0.4),              e1.g_vectors, 0))
twcl.add_layer("Strans",   ExtendedLayer(etw, Layer.half_infinite(e2, "transmission", 1), e1.g_vectors, 0))

'''
    Define the device and solve.
'''
device = ["Scylup", "Sair","Scyldo"]
twcl.set_stacking(device)
twcl.set_source(1/frequency, polarization[0], polarization[1], theta, phi)
twcl.solve()


if False:
    '''
        Get those fields in longitudinal plane.
    '''

    x, y, z = coords(0, 12, 0.5, 0.5, 1e-7, 1.5, (256, 1, 64))
    E, H = twcl.fields_volume2(x, y, tqdm(z)) # Progress bar on z axis
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
    plt.savefig("twited_fields.png")


if True:
    '''
        Get those fields in transversal plane
    '''
    #etw.plot()
    depth = 0.705
    x, y, z = coords(-10, 10, -10, 10, depth, depth, (512, 512, 1))
    E, H = twcl.fields_volume2(x, y, z) # Progress bar on z axis
    E = np.squeeze(E)
    print(E.shape)
    Exxy = (E[0]).real
    Eyxy = (E[1]).real
    Edisp = Exxy**2+Eyxy**2
    vmax = np.max(np.abs(Edisp))
    fig, ax = plt.subplots(figsize=(10,10))
    #ax.matshow(Edisp, cmap="RdBu", origin="lower", vmin=-vmax, vmax=vmax, extent=[ 0, np.max(x), 0, np.max(y)])
    ax.matshow(Edisp, cmap="inferno", origin="lower", vmin=0, vmax=vmax, extent=[ np.min(x), np.max(x), np.min(y), np.max(y)])
    ax.axis("equal")
    plt.savefig("twisted_fields_transversal.png")
