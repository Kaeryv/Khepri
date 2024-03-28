import sys
sys.path.append(".")
from bast.crystal import Crystal
from bast.draw import Drawing
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from bast.constants import c
import logging
#logging.getLogger().setLevel(logging.DEBUG)

# Define the structure
N = 512
d = Drawing((N, N), 12)
d.circle((0.0, 0.0), 0.4, 1.0)

# Create the crystal
pp = 7
cl = Crystal((pp,pp))
sc=2
cl.add_layer_uniform("S1", 1, 1.1/sc)
#cl.add_layer_uniform("Scyl", 1, 1.1/sc)
cl.add_layer_pixmap("Scyl", d.canvas(), 0.55/sc)
#cl.set_stacking(["Scyl", "S1", "Scyl"])
stacking = []
stacking.extend(["Scyl"]*sc)
stacking.extend(["S1"]*sc)
stacking.extend(["Scyl"]*sc)
cl.set_stacking(stacking)
#cl.set_stacking(["S1", "S1", "S1"])
d.plot("test.png")


# Compute spectrum
if False:
    chart = list()
    freqs = np.linspace(0.49, 0.6, 151)
    for f in freqs:
        wl = 1 / f
        cl.set_source(1 / f, 1.0, 0, 0.0, 0.0)
        cl.solve()
        chart.append(cl.poynting_flux_end())
    plt.figure()
    plt.plot(freqs, np.asarray(chart)[:, 1])
    plt.axvline(0.5185, color="k")
    d= np.loadtxt("data/holey_pair_45.csv", skiprows=7, delimiter=",")
    plt.plot(d[:,0]*1e12 / c * 1e-6, d[:,1])
    plt.savefig("Spectra_holey_pair.png")

# Compute field maps
if False:
    wl = 1.428
    cl.set_source(wl, 1.0, 0.0, 0, 0)
    cl.solve()
    start=0.0001
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    E, H = cl.fields_volume(tqdm(np.linspace(start, cl.stack_positions[-1], 64)))
    #E, H = cl.fields_volume2(x, y, tqdm(np.linspace(start, cl.stack_positions[-1], 64)))
    Exxz = E[:,0,:, 64].real
    Eyxz = E[:,1,:, 64].real
    Exyz = E[:,0,64, :].real
    Eyyz = E[:,1,64, :].real
    fig, axs = plt.subplots(2, 2, figsize=(4,6))
    axs = axs.flatten()
    for i, e in enumerate([Exxz, Eyxz, Exyz, Eyyz]):
        mx = np.max(e.real)
        axs[i].set_title(f"{i}")
        axs[i].matshow(e.real, origin="lower",cmap="RdBu", extent=[0, 1, start, cl.stack_positions[-1]], vmax=mx, vmin=-mx)
    #for z in cl.stack_positions[1:]:
    #    ax1.axhline(z, color="w", alpha=0.5)
    #    ax2.axhline(z, color="w", alpha=0.5)
    plt.savefig(f"Efield_debug.png", transparent=True)

# Compute field maps faster
if True:
    wl = 1/0.4 #1.428
    cl.set_source(wl, 1.0, 1.0, 0, 0)
    cl.solve()
    start=0.0001
    x = np.linspace(0, 1, 100)
    y = np.ones(100)*0.5
    E, H = cl.fields_volume2(x, y, tqdm(np.linspace(start, cl.stack_positions[-1], 64)))
    Exxz = E[:, 0].real
    Eyxz = E[:, 1].real
    E, H = cl.fields_volume2(y, x, tqdm(np.linspace(start, cl.stack_positions[-1], 64)))
    Exyz = E[:, 0].real
    Eyyz = E[:, 1].real
    fig, axs = plt.subplots(2, 2, figsize=(4,6))
    axs = axs.flatten()
    for i, e in enumerate([Exxz, Eyxz, Exyz, Eyyz]):
        mx = np.max(e.real)
        axs[i].set_title(f"{i}")
        axs[i].matshow(e.real, origin="lower",cmap="RdBu", extent=[0, 1, start, cl.stack_positions[-1]], vmax=mx, vmin=-mx)
    #for z in cl.stack_positions[1:]:
    #    ax1.axhline(z, color="w", alpha=0.5)
    #    ax2.axhline(z, color="w", alpha=0.5)
    plt.savefig(f"Efield_debug2.png", transparent=True)
