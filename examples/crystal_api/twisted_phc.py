import sys
sys.path.append(".")

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['BAST_MT_ON'] = '0'

from functools import partial
from multiprocessing import Pool
from bast.crystal import Crystal
from bast.draw import Drawing
from bast.expansion import Expansion
from bast.layer import Layer
import matplotlib.pyplot as plt
import numpy as np
from bast.crystal import Crystal
from tqdm import tqdm
from bast.misc import coords
from bast.tools import rotation_matrix as rot
from glob import glob
from itertools import product

'''
    Define the crystal layers. 
    Be careful that layers with different twist angles are different objects.
'''

def worker(config, **kwargs):
    wl, a = config
    return solve_crystal(wl, a, **kwargs)

def solve_crystal(wl, twist_angle, polar=(1,1), theta=0, phi=0, pw=(3,3), fields=False, return_rt=False):
    lattice = Crystal((1,1), lattice="hexagonal").lattice
    pattern = Drawing((256,256), 4, lattice=lattice)
    pattern.circle((0,0), 0.3, 1)
    e1 = Expansion(pw, lattice=lattice)
    e2 = Expansion(pw, lattice=lattice)
    e2.rotate(twist_angle)

    twcl = Crystal.from_expansion(e1+e2)
    
    etw = twcl.expansion
    twcl.add_layer("Sref",   Layer.half_infinite(e1, "reflexion", 1), True)
    twcl.add_layer("Scup",   Layer.analytical(e1, pattern.islands(), 4, 0.3), True)
    twcl.add_layer("Sair",   Layer.uniform(etw, 1.0, 0.3), False)
    twcl.add_layer("Scdo", Layer.analytical(e2, pattern.islands(), 4, 0.3), True)
    twcl.add_layer("Strans", Layer.half_infinite(e2, "transmission", 1), True)
    twcl.layers["Sair"].fields = True
    device = ["Scup", "Sair", "Scdo"]
    if fields:
        twcl.add_layer("Sbuffer",   Layer.uniform(etw, 1.0, 0.9), False)
        twcl.layers["Sbuffer"].fields = True
        device.extend(["Sbuffer"]*5)
        twcl.set_device(device, [True]*len(device))
    else:
        twcl.set_device(["Scup", "Sair", "Scdo"], [False]*3)
    twcl.set_source(wl, polar[0], polar[1], theta, phi)
    twcl.solve()
    if return_rt:
        return twcl.poynting_flux_end()
    else:
        return twcl

if __name__ == '__main__':
    if sys.argv[1] == "RT":
        M = 60
        RT = np.zeros((M,2))
        angle = float(sys.argv[3])
        for i, f in enumerate(np.linspace(0.7, 0.83, M)):
            twcl = solve_crystal(1/f, angle, pw=(3,3))
            RT[i] = twcl.poynting_flux_end()
    
        fig, ax = plt.subplots()
        ax.plot(RT[:, 0], label="R")
        ax.plot(RT[:, 1], label="T")
        fig.savefig(sys.argv[2])
        if len(sys.argv) >= 5:
            np.save(sys.argv[4], RT)
    
    if sys.argv[1] == 'twistmap':
        NF, NA = 200, 90 
        angles = np.linspace(0.0, 45, NA)
        freqs = np.linspace(0.78, 0.93, NF)
        RT = list()
        configs = list(product(1/freqs, angles))
        workerx = partial(worker, return_rt=True)
        with Pool(8) as p:
            for e in tqdm(p.imap_unordered(workerx, configs), total=len(configs) ):
                RT.append(e)
        RT = np.asarray(RT).reshape(NF, NA, 2)
    
        fig, ax = plt.subplots()
        im = ax.matshow(RT[:, :, 1])#, vmin=0, vmax=1)
        plt.colorbar(im)
        fig.savefig(sys.argv[2])
        if len(sys.argv) >= 5:
            np.save(sys.argv[3], RT)
    
    if sys.argv[1] == "map":
        pattern = sys.argv[2]
        files = glob(pattern)
        num_angles = len(files)
        rtmap = list()
        for i in range(num_angles):
            rtmap.append(np.load(pattern.replace("*", str(i))))
        rtmap = np.asarray(rtmap).reshape(num_angles, -1, 2)
        fig, ax = plt.subplots()
        ax.matshow(rtmap[..., 0].T, origin="lower", vmin=0, vmax=1)
        fig.savefig(sys.argv[3])
    
    
    if sys.argv[1] == "lg_fields":
        '''
            Get those fields in longitudinal plane.
        '''
        pwx = int(sys.argv[3])
        wl = float(eval(sys.argv[4]))
        ta = float(eval(sys.argv[5]))
        theta = 0
        twcl = solve_crystal(wl, ta, polar=(1,1), pw=(pwx,pwx), fields=True, theta=theta, phi=0)
        print("Solved crystal")
        y0 = 0.1
        x, y, z = coords(0, 24, y0, y0, -2, 4.5, (256, 1, 128))
        E, H = twcl.fields_volume(y, x, tqdm(z)) # Progress bar on z axis
        E = np.squeeze(E)
    
        '''
            Plot them.
        '''
        Exxz = E[:, 0, :]
        Eyxz = E[:, 1, :]
        Edisp = Exxz - 1j * Eyxz
        #Edisp = Ercp.real
        Emag = np.abs(Edisp)
        Ereal = Edisp.real
        vmax = np.max(np.abs(Edisp))
        fig, (ax1,ax2) = plt.subplots(2, figsize=(10,4.5))
        ax1.matshow(Ereal, cmap="bwr", origin="lower", vmin=-vmax, vmax=vmax, aspect=2.5, extent=[ 0, np.max(x), np.min(z), np.max(z)])
        ax2.arrow(np.max(x)/2, np.min(z) / 2, 0.5*np.sin(np.deg2rad(theta)), 0.5*np.cos(np.deg2rad(theta)), color="w", 
                  head_width=0.5, head_length=0.2)
        ax2.matshow(Emag, cmap="magma", origin="lower", vmin=0, aspect=2.5, extent=[ 0, np.max(x), np.min(z), np.max(z)])
    
    
        '''
            Overlay the structure (WIP).
        '''
        for z in [0.0, 0.2, 0.5, 0.7]:
            ax1.axhline(z, color="k")
            ax2.axhline(z, color="k")
    
        plt.xlabel("x[µm]")
        plt.ylabel("z[µm]")
        ax1.set_position([0.5, 0.05, 0.45, 0.9])
        ax2.set_position([0.05, 0.05, 0.45, 0.9])
        R, T = twcl.poynting_flux_end()
        T = T.real
        ax2.text(0.5, 3.0, f"T={T*100:0.0f}%", color="w")
        ax2.text(0.5, 3.5, f"twist={ta:0.2f} deg", color="w")
        ax2.text(0.5, 4.0, f"f={1/wl:0.2f}", color="w")
        plt.savefig(sys.argv[2])
    
    
    if sys.argv[1] == "tr_fields":
        pwx = int(sys.argv[3])
        wl = float(eval(sys.argv[4]))
        ta = float(eval(sys.argv[5]))
        twcl = solve_crystal(wl, ta, polar=(-1,1), pw=(pwx,pwx), fields=True)
        depth = 4
        w = 12
        x, y, z = coords(-w, w, -w, w, depth, depth, (512, 512, 1))
        E, H = twcl.fields_volume(x, y, z)
        E = np.squeeze(E)
        Edisp =  E[0] -1j*E[1] #**2+Eyxy**2#+Ezxy**2
    
        Emag = np.abs(Edisp)
        vmax = np.max(np.abs(Edisp.real))
        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(10,4.5))
        im = ax1.matshow(Emag, cmap="magma", origin="lower",  extent=[ np.min(x), np.max(x), np.min(y), np.max(y)])
        alpha = np.abs(Edisp) / np.max(np.abs(Edisp))
        im2 = ax2.matshow(np.angle(Edisp), cmap="hsv", origin="lower", vmin=-np.pi, vmax=np.pi, extent=[ np.min(x), np.max(x), np.min(y), np.max(y)], alpha=alpha)
        plt.colorbar(im)
        plt.colorbar(im2)
        ax1.axis("square")
        ax2.axis("square")
        R, T = twcl.poynting_flux_end()
        T = T.real
        ax1.text(0.5, 1.0, f"T={T*100:0.0f}%", color="w")
        ax1.text(0.5, 3.0, f"twist={ta:0.2f} deg", color="w")
        ax1.text(0.5, 5.0, f"f={1/wl:0.2f}", color="w")
        if sys.argv[2] != "n":
            plt.savefig(sys.argv[2])
