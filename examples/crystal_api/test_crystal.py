import sys
sys.path.append(".")
from bast.crystal import Crystal
from bast.draw import Drawing
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Define the structure
N = 512
d = Drawing((N, N), 12)
d.circle((0.0, 0.0), 0.4, 1.0)

# Create the crystal
pp = 7
cl = Crystal((pp,pp))
cl.add_layer_uniform("S1", 1, 1.1)
cl.add_layer_pixmap("Scyl", d.canvas(), 0.55)
cl.set_stacking(["Scyl", "S1", "Scyl"])
d.plot("test.png")


# Compute spectrum
if True:
    chart = list()
    freqs = np.linspace(0.49, 0.6, 101)
    for f in freqs:
        wl = 1 / f
        cl.set_source(1 / f, 1.0, 1.0, 0.0, 0.0)
        cl.solve()
        chart.append(cl.poynting_flux_end())

    plt.plot(freqs, np.asarray(chart)[:, 1])
    plt.savefig("Spectra_holey_pair.png")

# Compute field maps
if True:
    f = 1/1.80842
    wl = 1/f
    cl.set_source(wl, 1.0, 1.0, 0.0, 0.0)
    cl.solve()
    start=-0.2
    E, H = cl.fields_volume(tqdm(np.linspace(start, cl.stack_positions[-1], 64)))
    enormY = np.sqrt(E[:,1,64,:]**2)
    enormX = np.sqrt(E[:,0,64,:]**2)
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(4,6))
    ax1.matshow(enormY.real, origin="lower",cmap="hot", extent=[0, 1, start, cl.stack_positions[-1]])
    ax2.matshow(enormX.real, origin="lower",cmap="hot", extent=[0, 1, start, cl.stack_positions[-1]])
    for z in cl.stack_positions[1:]:
        ax1.axhline(z, color="w", alpha=0.5)
        ax2.axhline(z, color="w", alpha=0.5)
    plt.savefig(f"Efield_{f}.png", transparent=True)
