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
pw = (5,1)
N = 256
polarization = (1, -1j) # norm
theta = 0
phi = 0

'''
    Define the pattern (common to both layers)
'''
canvas_size = (N,1)
pattern = Drawing(canvas_size, 4)
pattern.rectangle((0,0), (0.5, 1), 1.0)
pattern.plot("figs/Woodpile_Pattern.png")

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
    twcl.add_layer("Scylup",   ExtendedLayer(etw, Layer.pixmap(e1, pattern.canvas(), 0.2),    e2.g_vectors, 1))
    twcl.add_layer("Sair",     ExtendedLayer(etw, Layer.uniform(e1, 1.0, 0.3),                e2.g_vectors, 1))
    twcl.add_layer("Scyldo",   ExtendedLayer(etw, Layer.pixmap(e2, pattern.canvas(), 0.2),    e1.g_vectors, 0))
    twcl.add_layer("Strans",   ExtendedLayer(etw, Layer.half_infinite(e2, "transmission", 1), e1.g_vectors, 0))

    '''
        Define the device and solve.
    '''
    device = ["Scylup","Scyldo"]
    twcl.set_stacking(device)
    twcl.set_source(1/frequency, polarization[0], polarization[1], theta, phi)
    twcl.solve()

    return twcl.poynting_flux_end()

NF=101
NA = 100
if sys.argv[1] == "c":
    frequencies = np.linspace(0.7, 0.98, NF)
    angles = np.linspace(0, 90, NA)
    from itertools import product
    RT = list()
    for i, (f, a) in enumerate(product(frequencies, angles)):
        RT.append(solve_rt(f, a))
    np.savez_compressed("woodpile2.npz", RT=RT, NA=NA, NF=NF)
elif sys.argv[1] == "p":
    RT = np.load(sys.argv[2])["RT"]
    R = np.reshape(RT, (NF, NA, 2))[...,0]
    R = np.reshape(RT, (NF, NA, 2))[...,1]
    fig, ax = plt.subplots()
    ax.matshow(R.real, origin="lower")
    fig.savefig("figs/Woodpile_R.png")
