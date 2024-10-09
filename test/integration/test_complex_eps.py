import unittest
# import sys
# sys.path.append(".")

from khepri import Crystal
import numpy as np
c= 299792458
covera = c / 1e-6
e0 = 8.85418782e-12
sigma = 0.01e6 # S/m
h = 1.2
from numpy.testing import assert_allclose
class TestBasicComplex(unittest.TestCase):
    def test_crystal(self):
        RT = list()
        wls = np.linspace(0.7, 2.0)
        for wl in wls:
            omega = np.pi * 2 * covera / wl 
            eps = 1.6**2 - 1j * sigma / omega / e0
            cl = Crystal((1,1))
            cl.add_layer_uniform("1", eps, h)
            cl.set_device(["1"])
            cl.set_source(wl, 1, 1)
            cl.solve()
            RT.append(cl.poynting_flux_end())
        RT = np.array(RT)
        
        omega = np.pi * 2 * covera / wls
        eps = 1.6**2 + 1j * sigma / omega / e0
        n1 = 1
        n2 = np.conj(np.sqrt(eps))
        r12 = (n1 - n2)/ (n1 + n2)
        r23 = (n2 - n1)/ (n2 + n1)
        rfresnel = np.abs((r12 + r23 * np.exp(-2j*2*np.pi/wls*n2*h)) / (1+r12*r23*np.exp(-2j*2*np.pi/wls*n2*h)))**2
        # import matplotlib.pyplot as plt
        # plt.plot(wls, np.array(RT)[:, 0], label="Khepri")
        # plt.plot(wls, rfresnel, "r:", label="Fresnel + thin film")
        # plt.legend()
        # plt.ylabel("R")
        # plt.xlabel("wavelength")
        # plt.show()  
        assert_allclose(rfresnel, RT[:,0])
if __name__ == '__main__':
    unittest.main()
