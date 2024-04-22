import sys
sys.path.append(".")

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['BAST_MT_ON'] = '0'

from bast.crystal import Crystal
from bast.draw import Drawing
from bast.constants import c
from bast.misc import coords

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from itertools import product
from bast.beams import gen_bzi_grid
from bast.beams import shifted_rotated_fields, _paraxial_gaussian_field_fn
from bast.beams import amplitudes_from_fields

import logging
# If you want more info
#logging.getLogger().setLevel(logging.DEBUG)

from multiprocessing import Pool
lattice = "square"
pp = 13
n1 = 2.02
r  = 0.368
depth = 0.316
pol = (1.0, 1j)
fmin, fmax = 0.57, 0.88
ffields = 0.715

# lattice = "hexagonal"
# pp = 3
# n1 = 2.02
# r  = 0.312
# depth = 0.118
# pol = (1.0, 1j)
# fmin, fmax = 0.85, 1.13

def solve(conf, return_crystal=False, fields=False, slicing=1):
    wl, kp = conf

    N = 512
    d = Drawing((N, N), n1**2)
    d.circle((0.0, 0.0), r, 1.0)
    #d.plot("test.png")

    cl = Crystal((pp,pp), lattice=lattice)
    cl.add_layer_analytical("1", d.islands(), n1**2, depth/slicing)
    stacking = []
    if fields:
        cl.add_layer_uniform("2", 1, 1.0/slicing)
        stacking.extend(["2"]*slicing)

    stacking.extend(["1"]*slicing)

    if fields:
        stacking.extend(["2"]*(13*slicing))
        
    cl.set_device(stacking, [fields]*len(stacking))
    
    cl.set_source(wl, *pol, 0.0, 0.0, kp=kp)
    cl.solve()
    if return_crystal:
        return cl.poynting_flux_end(), cl
    else:
        return cl.poynting_flux_end()


import psutil
def transverse_fields(kp, wl, bzi, NS, theta=0):
    x, y, _ = coords(0, bzi, 0, bzi, 0, 0, (NS*bzi, NS*bzi, 1))
    z = np.zeros_like(x)
    source_real = shifted_rotated_fields(
        _paraxial_gaussian_field_fn, x, y, z, wl, np.max(x)/2, np.max(y)/2, 1+depth/2, theta, 0, 0, beam_waist=3)
    source_real = np.asarray(source_real)
    source_real = np.swapaxes(source_real, 0, 2)
    source_real = np.swapaxes(source_real, 1, 3)
    
    source_real[...,0, 1] = source_real[...,0, 0] * 1j
    #source_real[...,1, 0] = source_real[...,0, 1]
    source_real[...,1, 1] = source_real[...,1, 0] * 1j
    #print(source_real.shape)
    #exit()

    _, cl = solve((wl, kp), return_crystal=True, fields=True, slicing=1)
    F = amplitudes_from_fields(source_real, cl.expansion, wl, kp, x, y, (bzi, bzi), a=1).flatten()
    E, _ = cl.fields_coords_xy(x, y, cl.zmax+12.1, F, use_lu=False)
    return E

def main(pp, zres, progress=True):
  
    if False:
        M = 100
        L = 201
        fs = np.linspace(fmin, fmax, L)
        wls = 1/fs
        bz_path = [ (np.pi *  kx, np.pi *  kx) for kx in np.linspace(0.35, 0, M)]
        bz_path.extend([(0.0, np.pi *  kx) for kx in np.linspace(0, 0.35, M)])
        configs = list(product(wls, bz_path))

        with Pool(12) as p:
            chart = list()
            for e in tqdm(p.imap_unordered(solve, configs), total=len(configs) ):
                chart.append(e)
        
        chart = np.asarray(chart).reshape(L, 2*M, 2)
        T = chart[:,:, 1]
        plt.figure()
        extent = [-np.max(bz_path), np.max(bz_path), np.min(fs), np.max(fs)]
        plt.matshow(T, vmin=0, vmax=1, cmap="Greys_r", origin='lower', extent=extent, aspect=4)
        plt.axhline(ffields)
        plt.savefig(f"BandDiagram_{lattice}.png")

    if False:
        wl = 1/ffields #1.428
        RT, cl = solve((wl, (0,0.07*np.pi)), return_crystal=True, fields=True, slicing=3)
        x, y, z = coords(0, 1, 0.25, 0.25, -1.4, cl.zmax+1.4, (100, 1, zres))
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
        mx = 0
        for i, e in enumerate([Exxz, Eyxz, Exyz, Eyyz]):
            mx = max(mx, np.max(np.abs(e.real)))

        for i, e in enumerate([Exxz, Eyxz, Exyz, Eyyz]):
            e = np.squeeze(e)
            axs[i].set_title(f"{i}")
            axs[i].matshow(np.tile(e.real, (1, 3)), origin="lower",cmap="RdBu", extent=extent, vmax=mx, vmin=-mx)
        for ax in axs:
            for z in cl.stack_positions[1:-1]:
                #ax.axhline(z, color="w", alpha=0.5)
                pass
        plt.savefig(f"Efield_bic.png", transparent=False)

    # The fields in the transversal plane
    # This is in order to obserbe the vortex at gamma BIC
    if True:
        from functools import partial

        wl = 1 / ffields #1.428
        bzi, NS = 45, 31
        kbz = gen_bzi_grid((bzi, bzi), a=1, reciproc=None).reshape(2, -1).T
        
        worker = partial(transverse_fields, wl=wl, bzi=bzi, NS=NS, theta=0)
        configs = [kp for kp in kbz]

        E_integrated = np.zeros((3, NS*bzi, NS*bzi), dtype='complex')
        with Pool(8) as p:
            fields = list()
            for e in tqdm(p.imap_unordered(worker, configs), total=len(configs) ):
                E_integrated += e

        fig, axs = plt.subplots(4, 2, figsize=(5,10))
        ecircular = E_integrated[0] - 1j * E_integrated[1]
        axs[0,0].matshow(np.abs(ecircular), cmap="hot")
        axs[0,1].matshow(np.angle(ecircular), cmap="hsv", vmin=-np.pi, vmax=np.pi)
        ecircular = E_integrated[0] + 1j * E_integrated[1]
        axs[1,0].matshow(np.abs(ecircular), cmap="hot")
        axs[1,1].matshow(np.angle(ecircular), cmap="hsv", vmin=-np.pi, vmax=np.pi)

        axs[2,0].matshow(np.abs(E_integrated[0]), cmap="hot", vmax=15)
        axs[2,1].matshow(np.angle(E_integrated[0]), cmap="hsv", vmin=-np.pi, vmax=np.pi)

        axs[3,0].matshow(np.abs(E_integrated[1]), cmap="hot", vmax=15)
        axs[3,1].matshow(np.angle(E_integrated[1]), cmap="hsv", vmin=-np.pi, vmax=np.pi)
        fig.savefig("BIC_PROFILE.png")



if __name__ == '__main__':
    main(9, 128)


# Compute spectrum
#if False:
#    chart = list()
#    freqs = np.linspace(0.49, 0.6, 151)
#    for f in tqdm(freqs):
#        cl.set_source(1 / f, 1.0, 0, 0.0, 0.0)
#        cl.solve()
#        chart.append(cl.poynting_flux_end())
#    plt.figure()
#    plt.plot(freqs, np.asarray(chart)[:, 1])
#    plt.axvline(0.5185, color="k")
#    plt.savefig("Spectra_holey_pair.png")
