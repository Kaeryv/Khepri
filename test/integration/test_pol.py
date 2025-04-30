import unittest
import numpy as np
from khepri import Crystal
import numpy as np
from scipy.io import loadmat
from numpy.testing import assert_allclose

pw = (3, 3)
M = 20
fixtures = "./test/integration/fixtures/"

class PolTest(unittest.TestCase):
    def test_scattering(self):
        cl  = Crystal(pw)
        cl.add_layer_uniform("U", 4, 0.3)
        cl.set_source(1.1, 1, 0)
        cl.set_device(["U"])
        cl.solve()
        x = y = np.array([[0]])
        z = x[0]
        E, H = cl.fields_volume(x, y, z)
        print(E.ravel())
