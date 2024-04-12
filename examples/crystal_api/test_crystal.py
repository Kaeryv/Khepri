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
def main(pp, zres, progress=True):
    N = 512
    d = Drawing((N, N), 12)
    d.circle((0.0, 0.0), 0.4, 1.0)

    # Create the crystal
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
    cl.set_device(stacking, [True]*len(stacking))
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

    from bast.misc import coords
    if True:
        wl = 1/0.4 #1.428
        cl.set_source(wl, 1.0, 1.0, 0, 0)
        cl.solve()
        x, y, z = coords(0, 1, 0.5, 0.5, 0.0001, cl.stack_positions[-2], (100, 1, zres))
        z = tqdm(z) if progress else z
        E, H = cl.fields_volume(x, y, z)
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
        #for z in cl.stack_positions[1:-1]:
        #    ax1.axhline(z, color="w", alpha=0.5)
        #    ax2.axhline(z, color="w", alpha=0.5)
        plt.savefig(f"Efield_debug2.png", transparent=True)


if __name__ == '__main__':
    main(7, 128)