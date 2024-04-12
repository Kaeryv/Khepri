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
pw = (3,1)
N = 256
polarization = (1, -1j) # norm
theta = 0
phi = 0

'''
    Define the pattern (common to both layers)
'''
canvas_size = (N,1)
pattern = Drawing(canvas_size, 1)
pattern.rectangle((0,0), (0.2, 1), 13.0)
pattern.plot("figs/Woodpile_Pattern.png")

pattern2 = Drawing(canvas_size, 1)
pattern2.rectangle((0,0), (0.2, 1), 13.0)
pattern2.plot("figs/Woodpile_Pattern2.png")

def solve_rt(frequency, angle_deg):
    e1 = Expansion(pw)
    e2 = Expansion(pw)
    e2.rotate(angle_deg)
    # e1.rotate(-angle_deg/2)

    '''
        Define the crystal layers. (Be careful that layers with different twist angles are different objects.)
    '''
    twcl = Crystal.from_expansion(e1+e2)
    etw = twcl.expansion
    twcl.add_layer("Sref",     ExtendedLayer(etw, Layer.half_infinite(e1, "reflexion", 1),    e2.g_vectors, 1))
    twcl.add_layer("x",   ExtendedLayer(etw, Layer.pixmap(e1, pattern.canvas(), 0.3),    e2.g_vectors, 1))
    twcl.add_layer("xp",   ExtendedLayer(etw, Layer.pixmap(e1, pattern2.canvas(), 0.3),    e2.g_vectors, 1))
    #twcl.add_layer("Sair",     ExtendedLayer(etw, Layer.uniform(e1, 1.0, 0.3),                e2.g_vectors, 1))
    twcl.add_layer("y",   ExtendedLayer(etw, Layer.pixmap(e2, pattern.canvas(), 0.3),    e1.g_vectors, 0))
    twcl.add_layer("yp",   ExtendedLayer(etw, Layer.pixmap(e2, pattern2.canvas(), 0.3),    e1.g_vectors, 0))
    twcl.add_layer("Strans",   ExtendedLayer(etw, Layer.half_infinite(e2, "transmission", 1), e1.g_vectors, 0))

    '''
        Define the device and solve.
    '''
    device = ["x","y", "x2", "y2", "x"] # , "Scylup"
    twcl.set_stacking(device)
    twcl.set_source(1/frequency, polarization[0], polarization[1], theta, phi)
    twcl.solve()

    return twcl.poynting_flux_end()

NF=101
NA = 40
if sys.argv[1] == "c":
    angles = np.linspace(45, 90, NA)
    frequencies = np.linspace(0.7, 0.98, NF)
    from itertools import product
    RT = list()
    for i, (f, a) in enumerate(tqdm(list(product(frequencies, angles)))):
        RT.append(solve_rt(f, a))
    np.savez_compressed("woodpile2.npz", RT=RT, F=frequencies, A=angles)
elif sys.argv[1] == "p":
    data = np.load(sys.argv[2])
    RT = data["RT"]
    frequencies = data["F"]
    angles = data["A"]
    NF = len(frequencies)
    NA = len(angles)
    R = np.reshape(RT, (NF, NA, 2))[...,0]
    T = np.reshape(RT, (NF, NA, 2))[...,1]
    fig, (ax1, ax2) = plt.subplots(2)
    extent = [np.min(angles), np.max(angles), np.min(frequencies), np.max(frequencies)]
    ax1.matshow(T.real, origin="lower", vmin=0, vmax=1, aspect=60, extent=extent)
    ax2.plot(frequencies, T[:, -1], color="r", label="Transmission")
    ax2.plot(frequencies, R[:, -1], color="b", label="Reflexion")
    plt.legend()
    fig.savefig("figs/Woodpile_R.png")
