import sys
sys.path.append(".")
assert(len(sys.argv) > 1)
action = sys.argv[1]

from bast.crystal import Crystal
from bast.extension import TwistedCrystal
from bast.tools import _joint_subspace, joint_subspace
from bast.draw import Drawing
import numpy as np
from bast.layer import Layer
from bast.expansion import Expansion
from bast.alternative import redheffer_product, incident, poynting_fluxes
import logging
#logging.getLogger().setLevel(logging.DEBUG)

pw = (3,3)
N = 256
canvas_size = (N,N)
pattern = Drawing(canvas_size, 4)
pattern.circle((0,0), 0.25, 1.0)


def extended_layer(layer: Layer, expansion: Expansion, wl, gs, mode):
    Ss = list()
    WIs = list()
    VIs = list()
    LIs = list()

    el = Layer()

    el.expansion = expansion

    for kp in gs.T:
        layer.solve(kp, wl)
        Ss.append(layer.S)
        WIs.append(layer.W)
        VIs.append(layer.V)
        LIs.append(layer.L)

    el.S =  joint_subspace(Ss, kind=mode)
    el.W = _joint_subspace(WIs, kind=mode)
    el.V = _joint_subspace(VIs, kind=mode)
    el.L = _joint_subspace(LIs, kind=mode)

    return el

from tqdm import tqdm
import matplotlib.pyplot as plt

def solve_rt(freq, angle_deg):
    k0 = 2*np.pi*freq
    
    # Upper untwisted layer
    e1 = Expansion(pw)
    layer_up = Layer.pixmap(e1, pattern.canvas(), 0.2)
    layer_air = Layer.uniform(e1, 1.0, 0.3)

    # Lower twisted layer at angle_deg
    e2 = Expansion(pw)
    e2.rotate(angle_deg)
    layer_down = Layer.pixmap(e2, pattern.canvas(), 0.2)
    
    # Extended basis and Smats
    etw = (e1+e2)
    l1 = extended_layer(layer_up,   etw, 1 / freq, e2.g_vectors, 1)
    li = extended_layer(layer_air, etw, 1/freq, e2.g_vectors, 1)
    l2 = extended_layer(layer_down, etw, 1 / freq, e1.g_vectors, 0)

    # Let's propagate a plane wave and get RT
    esrc = incident(etw.pw, 1, 0, k_vector=(0, 0, k0))
    Stot = redheffer_product(l1.S, li.S)
    Stot = redheffer_product(Stot, l2.S)
    S21, S11 = Stot[1,0], Stot[0,0]
    return poynting_fluxes(etw, S11 @ esrc, (0,0), 1/freq), poynting_fluxes(etw, S21 @ esrc, (0,0), 1/freq)

if action == "spectrum":
    freqs = np.linspace(0.7, 0.83, 100)
    assert(len(sys.argv)> 2)
    angle = float(sys.argv[2])
    spectrum = []
    for i, freq in enumerate(tqdm(freqs)):
        spectrum.append(solve_rt(freq, angle))
    plt.plot(spectrum)
    plt.savefig("debug.png")

elif action == "map":
    from bast.misc import str2linspace_args
    from itertools import product
    assert(len(sys.argv)> 3)
    angles = np.linspace(*str2linspace_args(sys.argv[2]))
    freqs = np.linspace(*str2linspace_args(sys.argv[3]))
    af = product(angles, freqs)
    spectrum = []
    for a, f in tqdm(list(af)):
        spectrum.append(solve_rt(f, a))
    plt.matshow(np.reshape(spectrum, (len(angles), len(freqs), 2))[:,:,1].T, vmin=0, vmax=1, origin="lower")
    np.savez_compressed("spectrum.npz", T=np.reshape(spectrum, (len(angles), len(freqs), 2))[:,:,1].T)
    plt.savefig("debug.png")
'''
elif action == "fields":
    assert(len(sys.argv)> 3)
    twist_angle = float(sys.argv[2])
    frequency = float(sys.argv[3])
    tc.solve(twist_angle, 1/frequency)
    tc.plot_lattice()
    zs = np.linspace(0,0.2,32)
    E3D = list()
    for z in zs:
        E, H = tc.fields_xy(z, 1/frequency)
        E3D.append(E)
    Enorm = np.linalg.norm(E3D, axis=1)
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.matshow(Enorm[:, :,64], cmap="hot")
    ax2.matshow(Enorm[:, 64,:], cmap="hot")
    fig.savefig("Figures/twisted_fields.png")

'''