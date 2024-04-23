import sys

sys.path.append(".")

from bast.crystal import Crystal
from bast.extension import ExtendedLayer
from bast.draw import Drawing
from bast.expansion import Expansion
from bast.layer import Layer
import matplotlib.pyplot as plt
import numpy as np
from bast.crystal import Crystal
from tqdm import tqdm
from bast.misc import coords
from bast.tools import rotation_matrix as rot
from bast.alternative import redheffer_product

from multiprocessing import Pool

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['BAST_MT_ON'] = '0'

"""
    Parameters
"""
N = 256
pwx = 3

polarization = (1, 1)  # norm
theta = 0
phi = 0

rods_height = 1.414 / 4
rods_shift = 0.5
rods_eps = 3.6**2
rods_w = 0.28

def solve_rt(frequency, angle_deg, kp=None):
    pw = (pwx, 1)
    canvas_size = (N, 1)
    pattern = Drawing(canvas_size, 1)
    pattern.rectangle((0, 0), (rods_w, 1), rods_eps)
    
    pattern2 = Drawing(canvas_size, 1)
    pattern2.rectangle((rods_shift, 0), (rods_w, 1), rods_eps)
    pattern2.rectangle((-rods_shift, 0), (rods_w, 1), rods_eps)
    e1 = Expansion(pw)
    e2 = Expansion(pw)
    e2.rotate(angle_deg)

    """
        Define the crystal layers. (Be careful that layers with different twist angles are different objects.)
    """
    twcl = Crystal.from_expansion(e1 + e2)
    etw = twcl.expansion
    twcl.add_layer("Sref",   Layer.half_infinite(e1, "reflexion", 1), True)
    twcl.add_layer("1",      Layer.pixmap(e1, pattern.canvas(),  rods_height), True)
    twcl.add_layer("2",      Layer.pixmap(e2, pattern.canvas(),  rods_height), True)
    twcl.add_layer("3",      Layer.pixmap(e1, pattern2.canvas(), rods_height), True)
    twcl.add_layer("4",      Layer.pixmap(e2, pattern2.canvas(), rods_height), True)
    twcl.add_layer("Strans", Layer.half_infinite(e2, "transmission", 1), True)

    """
        Define the device and solve.
    """
    device = ["1", "2", "3", "4"]
    twcl.set_device(device, [False] * len(device))
    twcl.set_source(1 / frequency, polarization[0], polarization[1], theta, phi, kp=kp)
    twcl.solve()

    return twcl.poynting_flux_end()



def worker_notwist(config):
    frequency, kp = config
    return solve_rt_notwist(frequency, kp=kp)

def solve_rt_notwist(frequency, kp=(0,0)):
    '''
        The wood pile structure without any twist.
        There are four layers in the z direction. 
        These could repeat to form a bigger crystal.
    ''' 
    pw = (pwx, pwx)
    pattern = Drawing((N, N), 1)
    pattern.rectangle((0, 0), (rods_w, 1), rods_eps)
    
    
    pattern2 = Drawing((N, N), 1)
    pattern2.rectangle(( rods_shift, 0), (rods_w, 1), rods_eps)
    pattern2.rectangle((-rods_shift, 0), (rods_w, 1), rods_eps)

    cl = Crystal(pw)
    cl.add_layer_pixmap("1", pattern.canvas(),    rods_height)
    cl.add_layer_pixmap("2", pattern.canvas().T,  rods_height)
    cl.add_layer_pixmap("3", pattern2.canvas(),   rods_height)
    cl.add_layer_pixmap("4", pattern2.canvas().T, rods_height)

    """
        Define the device and solve.
    """
    device = ['1','2','3','4']
    cl.set_device(device, [False] * len(device))
    cl.set_source(1 / frequency, polarization[0], polarization[1], 0, 0, kp=kp)
    cl.solve()
    cl.Stot = redheffer_product(cl.Stot, cl.Stot)
    cl.Stot = redheffer_product(cl.Stot, cl.Stot)

    return cl.poynting_flux_end()

if __name__ == '__main__':

    NF = 203
    NA = 90
    if sys.argv[1] == "ct":
        angles = np.linspace(45, 90, NA)
        frequencies = np.linspace(0.4, 0.65, NF)
        from itertools import product
    
        RT = list()
        for i, (f, a) in enumerate(tqdm(list(product(frequencies, angles)))):
            RT.append(solve_rt(f, a))
        np.savez_compressed("woodpile_twisted.npz", RT=RT, F=frequencies, A=angles)
    elif sys.argv[1] == "ctbd":
        angle  = 90
        frequencies = np.linspace(0.4, 0.65, NF)
        from itertools import product
        M = 11
        kpath = [ (kx, 0) for kx in np.linspace(0, 0.98*np.pi, M) ]
        kx = np.pi
        kpath.extend([(kx, ky) for ky in np.linspace(0, 0.98*np.pi, M)])
        kpath.extend([(kxy, kxy) for kxy in np.linspace(0.98*np.pi, 0, M)])
    
        RT = list()
        for i, (f, kp) in enumerate(tqdm(list(product(frequencies, kpath)))):
            RT.append(solve_rt(f, angle, kp=kp))
        np.savez_compressed("woodpile_twisted_bd2.npz", RT=RT, F=frequencies, A=kpath)
    elif sys.argv[1] == "c":
        frequencies = np.linspace(0.4, 0.65, NF)
        RT = list()
        for i, f in enumerate(tqdm(frequencies)):
            RT.append(solve_rt_notwist(f))
        np.savez_compressed("woodpile2.npz", RT=RT, F=frequencies)
    elif sys.argv[1] == "cbd":
        frequencies = np.linspace(0.4, 0.7, NF)
        from itertools import product
        M = 91
        kpath = [ (kx, 0) for kx in np.linspace(0, 0.99*np.pi, M) ]
    
        RT = list()
        with Pool(6) as p:
            configs = list(product(frequencies, kpath))
            #for i, (f, kp) in enumerate(tqdm():
            RT = p.map(worker_notwist, configs)
        np.savez_compressed("woodpile_untwisted_bd.npz", RT=RT, F=frequencies, A=kpath)
    elif sys.argv[1] == "pt":
        data = np.load(sys.argv[2])
        print(list(data.keys()))
        RT = data["RT"]
        frequencies = data["F"]
        angles = data["A"]
        NF = len(frequencies)
        NA = len(angles)
        R = np.reshape(RT, (NF, NA, 2))[..., 0]
        T = np.reshape(RT, (NF, NA, 2))[..., 1]
        fig, (ax1, ax2) = plt.subplots(2)
        extent = [np.min(angles), np.max(angles), np.min(frequencies), np.max(frequencies)]
        im = ax1.matshow(T.real, origin="lower", vmin=0, vmax=1, aspect=60/6, extent=extent, cmap="Greys_r")
        plt.colorbar(im)
        ax1.set_title("Twisting Woodpile")
        ax1.set_xlabel("Frequency [c/a]")
    
        ax2.plot(frequencies, T[:, 25], color="r", label="Transmission")
        ax2.plot(frequencies, R[:, 25], color="b", label="Reflexion")
        plt.legend()
        fig.savefig("figs/Woodpile_R.png")
    elif sys.argv[1] == "p":
        data = np.load(sys.argv[2])
        RT = data["RT"]
        frequencies = data["F"]
        NF = len(frequencies)
        R = np.reshape(RT, (NF, 2))[..., 0]
        T = np.reshape(RT, (NF, 2))[..., 1]
        fig, ax2 = plt.subplots()
        ax2.plot(frequencies, T[:], color="r", label="Transmission")
        ax2.plot(frequencies, R[:], color="b", label="Reflexion")
        ax2.set_title("Untwisted Woodpile")
        ax2.set_ylabel("Frequency [c/a]")
        plt.legend()
        fig.savefig("figs/Woodpile_R_notwist.png")
