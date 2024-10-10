import sys

sys.path.append(".")

from khepri.crystal import Crystal
from khepri.draw import Drawing
from khepri.expansion import Expansion
from khepri.layer import Layer
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from khepri.misc import coords
from khepri.alternative import redheffer_product

from multiprocessing import Pool

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['BAST_MT_ON'] = '0'

"""
    Parameters
    From 'A three-dimentional photonic crystal operating at infrared wavelengths
    SY. Lin and JG. Fleming. Letters to Nature 1998
"""
N = 512
pwx = 3

polarization = (1, 1)  # norm
theta = 0
phi = 0

rods_height = 1.414 / 4
rods_shift = 0.5
rods_eps = 3.6**2
rods_w = 0.28

def solve_rt(frequency, angle, kp=None):
    pw = (pwx, 1)
    canvas_size = (N, 1)
    pattern = Drawing(canvas_size, 1)
    pattern.rectangle((0, 0), (rods_w, 1), rods_eps)
    
    pattern2 = Drawing(canvas_size, 1)
    pattern2.rectangle((rods_shift, 0), (rods_w, 1), rods_eps)
    pattern2.rectangle((-rods_shift, 0), (rods_w, 1), rods_eps)
    e1 = Expansion(pw)
    e2 = Expansion(pw)
    e2.rotate(angle)

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



from khepri.factory import make_woodpile
def worker_notwist(config):
    frequency, kp = config
    wl = 1 / frequency
    cl = make_woodpile(rods_w, rods_eps, rods_shift, rods_height, (pwx, pwx))
    cl.set_source(wl, polarization[0], polarization[1], 0, 0, kp=kp)
    cl.solve()
    cl.Stot = redheffer_product(cl.Stot, cl.Stot)
    #cl.Stot = redheffer_product(cl.Stot, cl.Stot)

    return cl.poynting_flux_end()

if __name__ == '__main__':
    NF = 203
    NA = 90
    if sys.argv[1] == "ct":
        angles = np.deg2rad(np.linspace(45, 90, NA))
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
        a = 1.414
        freqs = np.linspace(0.35/a, 0.65/a, NF)
        kp = (0, 0)
        RT = list()
        for i, f in enumerate(tqdm(freqs)):
            RT.append(worker_notwist((f,kp)))
        np.savez_compressed("woodpile2.npz", RT=RT, F=a*freqs)
    elif sys.argv[1] == "cbd":
        a = 1.414
        frequencies = np.linspace(0.4/a, 0.65/a, NF)
        from itertools import product
        M = 91
        kpath = [ (kx, 0) for kx in np.linspace(0, 0.99*np.pi, M) ]
    
        RT = list()
        with Pool(8) as p:
            configs = list(product(frequencies, kpath))
            #for i, (f, kp) in enumerate(tqdm():
            RT = p.map(worker_notwist, configs)
        np.savez_compressed("woodpile_untwisted_bd.npz", RT=RT, F=frequencies, A=kpath)
    elif sys.argv[1] == "pt":
        data = np.load(sys.argv[2])
        print(list(data.keys()))
        RT = data["RT"]
        frequencies = data["F"] * 1.414
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
        tidy3D = np.loadtxt("test_woodpile.csv", skiprows=7, delimiter=",")
        data = np.load(sys.argv[2])
        RT = data["RT"]
        wls = data["F"]
        NF = len(wls)
        R = np.reshape(RT, (NF, 2))[..., 0]
        T = np.reshape(RT, (NF, 2))[..., 1]
        fig, ax2 = plt.subplots()
        ax2.plot(wls, T[:], color="r", label="Bast", marker='o')
        #ax2.plot(tidy3D[:,0], tidy3D[:,1], color="b", label="Tidy3D", marker='o')
        #ax2.plot(1/frequencies, R[:], color="b", label="Reflexion")
        ax2.set_title("Untwisted Woodpile")
        ax2.set_ylabel("Frequency [c/a]")
        plt.legend()
        fig.savefig("figs/Woodpile_R_notwist.png")
