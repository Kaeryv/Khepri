"""
    Testing the effective medium configuration for a cylinder.
    The structure is the one from **Molding the flow of Light**
    Comparison is made against validated matlab RCWA.
"""
import unittest

from bast.draw import Drawing
from numpy.testing import assert_allclose
import numpy as np
import matplotlib.pyplot as plt
N =13
epsh = 12
class DrawingTest(unittest.TestCase):
    def test_rectangle(self):
        d = Drawing((N, 1), epsh)
        d.rectangle((0.0, 0.0), (0.2, 1), 1)
        d.rectangle((-0.4, 0.0), (0.2, 1), 1)
        d.rectangle((0.4, 0.0), (0.2, 1), 1)
        d.rectangle((0.1, 0.0), (0.4, 1), 1)
        #plt.matshow(d.canvas())
        #plt.savefig("test.png")
        #assert_allclose(S.numpy(), valid)

    def test_from_numpy(self):
        d = Drawing((N, 1), epsh)
        d.rectangle((0.0, 0.0), (0.2, 1), 1)
        #d.rectangle((-0.4, 0.0), (0.2, 1), 1)
        #d.rectangle((0.4, 0.0), (0.2, 1), 1)
        d.rectangle((0.1, 0.0), (0.4, 1), 1)
        d.rectangle((0.25, 0.0), (0.2, 1), 1)
        d.rectangle((-0.25, 0.0), (0.1, 1), 1)
        canvas = d.canvas()

        d2 = Drawing((N, 1), 1)
        d2.from_numpy(canvas)
        assert_allclose(d.canvas(), d2.canvas())



