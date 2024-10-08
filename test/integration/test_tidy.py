import unittest

from bast.crystal import Crystal
from bast.draw import Drawing
import matplotlib.pyplot as plt
import numpy as np
from bast.crystal import Crystal

N = 512
pwx = 9
polarization = (1, 1)  # norm

def solve_rt_notwist(wl, kp=(0,0)):
    '''
    The wood pile structure without any twist.
    There are four layers in the z direction. 
    These could repeat to form a bigger crystal.
    ''' 
    pw = (pwx, pwx)
    pattern = Drawing((N, N), 1)
    pattern.rectangle((0, 0), (0.5, 0.5), 4)
    cl = Crystal(pw)
    cl.add_layer_analytical("1", pattern.islands(), pattern.background,  1.0)

    """
        Define the device and solve.
    """
    device = ['1']
    cl.set_device(device, [False] * len(device))
    cl.set_source(wl, polarization[0], polarization[1], 0, 0, kp=kp)
    cl.solve()
    return cl.poynting_flux_end()

class Tidy3DTests(unittest.TestCase):
    def test_square(self):
        NF = 13
        wls = np.linspace(1.01, 2, NF)
        RT = list()
        for wl in wls:
            RT.append(solve_rt_notwist(wl))
        tidy3D = np.loadtxt("./test/integration/fixtures/tidy3d_square.csv", skiprows=7, delimiter=",")
        T = np.reshape(RT, (NF, 2))[..., 1]
        fig, ax2 = plt.subplots()
        ax2.plot(wls, T[:], color="r", label="Bast (RCWA)", marker='o')
        ax2.plot(tidy3D[:,0], tidy3D[:,1], color="b", label="Tidy3D (FDTD)", marker='o')
        ax2.set_title("Testing for {NF} points with {pw=}")
        ax2.set_ylabel("Frequency [c/a]")
        plt.legend()
        fig.savefig("test/figures/test_tidy3D_square.png")
