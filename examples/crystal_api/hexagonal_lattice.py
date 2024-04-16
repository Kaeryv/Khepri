import sys

sys.path.append(".")
from bast.crystal import Crystal
from bast.draw import Drawing
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from bast.constants import c
import logging

# logging.getLogger().setLevel(logging.DEBUG)


from scipy.io import loadmat
data = loadmat('hexa_11_77.mat')
tmat_ttot = data["Ttot"]
tmat_lambda = data["lambda"] * 1e6

# Define the structure
def main(pp, zres, progress=True):
    cl = Crystal((pp,) * 2, lattice="hexagonal")
    N = 1024
    d = Drawing((N, N), 12, cl.lattice)
    d.circle((0.0, 0.0), 0.40, 1.0)
    d.plot("Structure.png")
    sl = 8
    cl.add_layer_pixmap("Scyl", d.canvas(shape=(256,256)), 0.55/sl)
    cl.add_layer_uniform("Si", 1.0, 1.1/sl)
    stacking = []
    stacking.extend(["Scyl"]*sl)
    stacking.extend(["Si"]*sl)
    stacking.extend(["Scyl"]*sl)

    cl.set_device(stacking, [True] * len(stacking))

    if True:
        chart = list()
        wls = np.linspace(1.46666, 1/0.49, 300)
        print(np.min(wls), np.max(wls))
        for wl in wls:
            cl.set_source(wl, 1.0, 1.0, 0.0, 0.0)
            cl.solve()
            chart.append(cl.poynting_flux_end())

        plt.figure()
        plt.plot(wls, np.asarray(chart)[:, 1])
        plt.plot(tmat_lambda, tmat_ttot, "r-")
        #plt.axvline(0.5185, color="k")
        # d = np.loadtxt("data/holey_pair_45.csv", skiprows=7, delimiter=",")
        # plt.plot(d[:, 0] * 1e12 / c * 1e-6, d[:, 1])
        plt.savefig("Spectra_holey_pair.png")

    from bast.misc import coords

    if False:
        wl = 1 / 0.4  # 1.428
        cl.set_source(wl, 1.0, 1.0, 0, 0)
        cl.solve()
        x, y, z = coords(0, 1, 0.5, 0.5, -1, cl.zmax, (100, 1, zres))
        zvals = tqdm(z) if progress else z
        E, _ = cl.fields_volume(x, y, zvals)
        Exxz = E[:, 0].real
        Eyxz = E[:, 1].real
        E, _ = cl.fields_volume(y, x, z)
        Exyz = E[:, 0].real
        Eyyz = E[:, 1].real
        fig, axs = plt.subplots(2, 2, figsize=(4, 6))
        axs = axs.flatten()
        extent = [np.min(x), np.max(x), np.min(z), np.max(z)]
        print(extent)
        for i, e in enumerate([Exxz, Eyxz, Exyz, Eyyz]):
            e = np.squeeze(e)
            mx = np.max(e.real)
            axs[i].set_title(f"{i}")
            axs[i].matshow(
                e.real, origin="lower", cmap="RdBu", extent=extent, vmax=mx, vmin=-mx
            )
        # for z in cl.stack_positions[1:-1]:
        #    ax1.axhline(z, color="w", alpha=0.5)
        #    ax2.axhline(z, color="w", alpha=0.5)
        fig.savefig(f"Efield_debug2.png", transparent=True)


if __name__ == "__main__":
    main(5, 128)
