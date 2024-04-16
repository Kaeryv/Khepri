import sys
sys.path.append(".")

from bast.crystal import Crystal
from bast.draw import Drawing
from bast.constants import c
from bast.misc import coords

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import logging
# If you want more info
#logging.getLogger().setLevel(logging.DEBUG)

def main(pp, zres, progress=True):
    N = 512
    d = Drawing((N, N), 12)
    d.circle((0.0, 0.0), 0.4, 1.0)
    d.plot("test.png")

    cl = Crystal((pp,pp))
    cl.add_layer_uniform("S1", 1, 1.1)
    cl.add_layer_pixmap("Scyl", d.canvas(), 0.55)
    stacking = ["Scyl", "S1", "Scyl"]
    cl.set_device(stacking, [True]*len(stacking))


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
        plt.savefig("Spectra_holey_pair.png")

    if True:
        wl = 1/0.4 #1.428
        cl.set_source(wl, 1.0, 1.0, 0, 0)
        cl.solve()
        x, y, z = coords(0, 1, 0.5, 0.5, 0.0001, cl.stack_positions[-2], (100, 1, zres))
        zvals = tqdm(z) if progress else z
        E, H = cl.fields_volume(x, y, zvals)
        Exxz = E[:, 0].real
        Eyxz = E[:, 1].real
        E, H = cl.fields_volume(y, x, z)
        Exyz = E[:, 0].real
        Eyyz = E[:, 1].real
        fig, axs = plt.subplots(2, 2, figsize=(4,6))
        axs = axs.flatten()
        extent = [np.min(x), np.max(x), np.min(z), np.max(z)]
        for i, e in enumerate([Exxz, Eyxz, Exyz, Eyyz]):
            e = np.squeeze(e)
            mx = np.max(e.real)
            axs[i].set_title(f"{i}")
            axs[i].matshow(e.real, origin="lower",cmap="RdBu", extent=extent, vmax=mx, vmin=-mx)
        for ax in axs:
            for z in cl.stack_positions[1:-1]:
                ax.axhline(z, color="w", alpha=0.5)

        plt.savefig(f"Efield_holey_pair.png", transparent=True)


if __name__ == '__main__':
    main(7, 128)
