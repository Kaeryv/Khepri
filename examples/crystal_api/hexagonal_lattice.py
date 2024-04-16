import sys

sys.path.append(".")
from bast.crystal import Crystal
from bast.draw import Drawing
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from bast.constants import c
from bast.fourier import transform_disc
import logging

# logging.getLogger().setLevel(logging.DEBUG)


from scipy.io import loadmat
data = loadmat('square_11_77.mat')
tmat_ttot = data["Ttot"]
tmat_lambda = data["lambda"] * 1e6

# Define the structure
def main(pp, zres, progress=True):
    cl = Crystal((pp,) * 2, lattice="square")
    N = 512
    radius = 0.4
    epsh = 12
    d = Drawing((N, N), epsh, cl.lattice)
    d.circle((0.0, 0.0), radius, 1)
    #fig, (ax1, ax2) = plt.subplots(2)
    #gx, gy = cl.expansion.g_vectors
    #ff = transform_disc(0., 0., radius, gx, gy, 1).reshape(pp,pp).T * epsh
    #ff = np.fft.ifft2(np.fft.ifftshift(ff)) * pp * pp
    #im1 = ax1.matshow(np.tile(ff.real, (2,2)), origin="lower")
    #plt.colorbar(im1)
    #c = (N-1) // 2
    ##ax2.set_xlim(c-pp//2, c+pp//2)
    ##ax2.set_ylim(c-pp//2, c+pp//2)
    #d.plot("Structure.png", what="reconstruct", ax=ax2, tiling=(2,2))
    sl = 2
    #cl.add_layer_pixmap("Scyl", d.canvas(), 0.55/sl)
    cl.add_layer_analytical("Scyl", d.islands(), epsh, 0.55/sl)
    cl.add_layer_uniform("Si", 1.0, 1.1/sl)
    stacking = []
    stacking.extend(["Scyl"]*sl)
    stacking.extend(["Si"]*sl)
    stacking.extend(["Scyl"]*sl)

    cl.set_device(stacking, [True] * len(stacking))
    tidy3D2 = np.loadtxt("holey_pair_30.csv", delimiter=",")
    tidy3D = np.loadtxt("holey_pair.csv", delimiter=",")
    if True:
        chart = list()
        wls = np.linspace(1.46666, 1/0.49, 300)
        print(np.min(wls), np.max(wls))
        for wl in wls:
            cl.set_source(wl, 1.0, 1.0, 0.0, 0.0)
            cl.solve()
            chart.append(cl.poynting_flux_end())

        fig, ax = plt.subplots()
        ax.plot(wls, np.asarray(chart)[:, 1], 'b-')
        ax.plot(tmat_lambda, tmat_ttot, "r-")
        #ax.plot(tidy3D[:,0], tidy3D[:,1], "r:")
        ax.plot(tidy3D2[:,0], tidy3D2[:,1], "k:")
        
        #plt.axvline(0.5185, color="k")
        # d = np.loadtxt("data/holey_pair_45.csv", skiprows=7, delimiter=",")
        # plt.plot(d[:, 0] * 1e12 / c * 1e-6, d[:, 1])
        fig.savefig("spectre.png")

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
