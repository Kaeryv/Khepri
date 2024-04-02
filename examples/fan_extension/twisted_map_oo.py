import sys
sys.path.append(".")
assert(len(sys.argv) > 1)
action = sys.argv[1]

from bast.crystal import Crystal
from bast.extension import TwistedCrystal
from bast.tools import _joint_subspace
from bast.draw import Drawing
import numpy as np
import logging
#logging.getLogger().setLevel(logging.DEBUG)

pw = (3,3)
N = 256
canvas_size = (N,N)
pattern = Drawing(canvas_size, 4)
pattern.circle((0,0), 0.25, 1.0)
#pattern.plot(shape=(100,100), interp_method="pchip", filename="pattern.png")

cup = Crystal(pw, void=True)
cup.add_layer_pixmap("S1", pattern.canvas(), 0.2)
cup.add_layer_uniform("Si", 1.0, 0.3)
cup.set_stacking(["S1", "Si"])
cdown = Crystal(pw, void=True)
cdown.add_layer_pixmap("S2", pattern.canvas(), 0.2)
cdown.set_stacking(["S2"])


tc = TwistedCrystal(cup, cdown)



spectrum = list()


from tqdm import tqdm
import matplotlib.pyplot as plt
if action == "spectrum":
    freqs = np.linspace(0.7, 0.83, 100)
    assert(len(sys.argv)> 2)
    angle = float(sys.argv[2])
    for i, freq in enumerate(tqdm(freqs)):
        tc.solve(angle, 1/freq)
        R, T = tc.poynting_fluxes_end()
        spectrum.append((R,T))
    plt.plot(spectrum)
    plt.savefig("debug.png")

elif action == "map":
    from bast.misc import str2linspace_args
    from itertools import product
    assert(len(sys.argv)> 3)
    angles = np.linspace(*str2linspace_args(sys.argv[2]))
    freqs = np.linspace(*str2linspace_args(sys.argv[3]))
    af = product(angles, freqs)
    for a, f in tqdm(list(af)):
        tc.solve(a, 1/f)
        T, R = tc.poynting_fluxes_end()
        spectrum.append((R,T))
    plt.matshow(np.reshape(spectrum, (len(angles), len(freqs), 2))[:,:,1].T, vmin=0, vmax=1, origin="lower")
    np.savez_compressed("spectrum.npz", T=np.reshape(spectrum, (len(angles), len(freqs), 2))[:,:,1].T)
    plt.savefig("debug.png")

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

    