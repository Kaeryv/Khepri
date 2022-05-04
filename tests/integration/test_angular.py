import unittest
import numpy as np

from bast.tools import nanometers
from bast.lattice import CartesianLattice
from bast.scattering import scattering_matrix
import numpy as np
from scipy.io import loadmat
from numpy.testing import assert_allclose

pw = (3, 3)
a  = nanometers(100)
wavelength = nanometers(200)
M = 20
fixtures = "./tests/integration/fixtures/"

thetas = np.linspace(0, 89.9, M)
class AngularTest(unittest.TestCase):
    def test_scattering(self):
        data = loadmat(f'{fixtures}/angular/Sstheta.mat')
        for i, theta in enumerate(thetas):
            lattice = CartesianLattice(pw, a1=(a, 0.0), a2=(0.0, a), eps_emerg=1.0, eps_incid=1.0, dtype=np.float64)
            kp = lattice.kp_angle(wavelength, theta, 0.0)
            S, _ = scattering_matrix(pw, lattice, "disc", [0.5*a, 0.5*a, 0.2*a], 8.9, 1.0, wavelength, kp, a, 3)
            assert_allclose(S, data["Ss"][i], atol=1e-10)
