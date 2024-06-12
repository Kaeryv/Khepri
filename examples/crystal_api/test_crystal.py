import sys
sys.path.append(".")

from bast.crystal import Crystal
from bast.draw import Drawing
from bast.constants import c
from bast.misc import coords, quiver, poynting_vector

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
        wl = 2.04 #1/0.4 #1.428
        cl.set_source(wl, 1.0, 0.0, 0, 0)
        cl.solve()
        xyres = 50
        x, y, z = coords(0, 1, 0.0, 1.0, 0.0001, cl.depth, (xyres, xyres, zres))
        cut = xyres // 2
        zvals = tqdm(z) if progress else z
        E, H = cl.fields_volume(x, y, zvals)
        
        Exxz = E[:, 0, :, cut].real
        Eyxz = E[:, 1, :, cut].real
        Exyz = E[:, 0, cut, :].real
        Eyyz = E[:, 1, cut, :].real

        fig, axs = plt.subplots(2, 2, figsize=(4,6))
        axs = axs.flatten()
        extent = [np.min(x), np.max(x), np.min(z), np.max(z)]
        names = ["Ex-xz", "Ey-xz", "Ex-yz","Ey-yz"]
        for i, (e, n) in enumerate(zip([Exxz, Eyxz, Exyz, Eyyz], names)):
            e = np.squeeze(e)
            mx = np.max(e.real)
            axs[i].set_title(n)
            im = axs[i].matshow(e.real, origin="lower",cmap="RdBu", extent=extent, vmax=mx, vmin=-mx)
            plt.colorbar(im)
            for zl in cl.stack_positions[1:-1]:
                axs[i].axhline(zl, color="k", alpha=0.5)

        fig.tight_layout()
        fig.savefig(f"Efield_holey_pair.png", transparent=True)
        P = poynting_vector(E, H, 1)
        Pyz = P[:, :, cut, :].real
        Pyznorm = np.linalg.norm(Pyz, axis=1)
        Pxz = P[:, :, :, cut].real
        Pxznorm = np.linalg.norm(Pxz, axis=1)

        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(4,6))
        im1 = ax1.matshow(Pyznorm, cmap="magma", origin="lower",vmin=0, extent=extent)
        yz1, yz2 = np.meshgrid(y[:,0], z)
        quiver(ax1, yz1, yz2, Pyz[:, 2], Pyz[:, 1], subsample=4, color="w")
        im2 = ax2.matshow(Pxznorm, origin="lower",cmap="magma", vmin=0, extent=extent)
        quiver(ax2, yz1, yz2, Pxz[:, 2], Pxz[:, 0] , subsample=4, color="w")
        ax1.set_title("Poynting-YZ")
        ax2.set_title("Poynting-XZ")
        fig.colorbar(im1)
        fig.colorbar(im2)
        fig.savefig(f"Poynting_holey_pair.png", transparent=True)

if __name__ == '__main__':
    main(7, 128)
