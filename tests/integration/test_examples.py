"""
    Testing the effective medium configuration for a cylinder.
    The structure is the one from **Molding the flow of Light**
    Comparison is made against validated matlab RCWA.
"""
import unittest
from numpy.testing import assert_allclose

class ExamplesTest(unittest.TestCase):
    def test_example(self):
        from examples.crystal_api.test_crystal import main
        main(3, 32, progress=False)



