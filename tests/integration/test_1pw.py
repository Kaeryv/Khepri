"""
    Testing the effective medium configuration for a cylinder.
    The structure is the one from **Molding the flow of Light**
    Comparison is made against validated matlab RCWA.
"""
import unittest
from bast.tools import nanometers
from bast.tmat.lattice import CartesianLattice
from bast.tmat.scattering import scattering_matrix

from numpy.testing import assert_allclose
from scipy.io import loadmat
import numpy as np

pw = (1, 1)
a  = nanometers(100)
wavelength = nanometers(200)
fixtures = "./tests/integration/fixtures/1pw/"

class EffectiveMediumTest(unittest.TestCase):
    def test_scattering(self):
        valid = loadmat(f'{fixtures}/S.mat')["Sslice"]
        lattice = CartesianLattice(pw, a1=(a, 0.0), a2=(0.0, a), eps_emerg=1.0, eps_incid=1.0, dtype=np.float64)
        S = scattering_matrix(pw, lattice, "disc", [0.5*a, 0.5*a, 0.2*a], 8.9, 1.0, wavelength, (0, 0), a, 3)
        assert_allclose(S.numpy(), valid)


