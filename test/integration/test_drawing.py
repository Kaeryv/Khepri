"""
    Testing the effective medium configuration for a cylinder.
    The structure is the one from **Molding the flow of Light**
    Comparison is made against validated matlab RCWA.
"""
import unittest

from bast.draw import Drawing
from bast.expansion import Expansion
from bast.fourier import transform, combine_fourier_masks
from numpy.testing import assert_allclose
import numpy as np
import matplotlib.pyplot as plt
N =51
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

    def test_compare_rectangle(self):
        d = Drawing((N, N), 1)

        d.rectangle((0,0), (1, 0.3), 8)
        fig, axs = plt.subplots(2, 2) 
        axs[0,0].matshow(d.canvas(), cmap='Blues')

        fft = np.fft.fftshift(np.fft.fft2(d.canvas()))
        axs[0,1].matshow(np.abs(fft), cmap='Blues')

        sigma = 1
        e = Expansion((N,N))
        Gx, Gy = e.g_vectors
        islands_data = [ (transform(isl["type"], isl["params"], Gx, Gy, sigma), isl["epsilon"]) for isl in d.islands() ]
        fourier = combine_fourier_masks(islands_data, 1, inverse=False).reshape((N,N))
        axs[1,1].matshow(np.abs(fourier), cmap='Blues')

        real = np.fft.ifft2(np.fft.ifftshift(fourier)) * N * N
        axs[1,0].matshow(np.real(real), cmap='Blues', vmin=1, vmax=8)

        plt.savefig('test_draw_rectangle.png')


    def test_compare_disc(self):
        d = Drawing((N, N), 1)

        d.circle((0,0), 0.3, 8)
        fig, axs = plt.subplots(2, 2) 
        axs[0,0].matshow(d.canvas(), cmap='Blues')

        fft = np.fft.fftshift(np.fft.fft2(d.canvas()))
        axs[0,1].matshow(np.abs(fft), cmap='Blues')

        #real = np.fft.ifft2(np.fft.ifftshift(fft)) * N * N
        #axs[0,0].matshow(np.real(real), cmap='Blues', vmin=1, vmax=8)

        sigma = 1
        e = Expansion((N,N))
        Gx, Gy = e.g_vectors
        islands_data = [ (transform(isl["type"], isl["params"], Gx, Gy, sigma), isl["epsilon"]) for isl in d.islands() ]
        fourier = combine_fourier_masks(islands_data, 1, inverse=False).reshape((N,N))
        axs[1,1].matshow(np.abs(fourier), cmap='Blues')

        real = np.fft.ifft2(np.fft.ifftshift(fourier)) * N * N
        axs[1,0].matshow(np.real(real), cmap='Blues', vmin=1, vmax=8)

        plt.savefig('test_disc.png')

    def test_compare_misc(self):
        d = Drawing((N, N), 1)
        d.rectangle((0,-0.4), (1, 0.2), 8)
        d.rectangle((0,-0.2), (1, 0.1), 8)
        d.rectangle((0,0.0), (1, 0.1), 8)
        d.circle((0,0.31), 0.2, 8)
        fig, axs = plt.subplots(2, 2) 
        axs[0,0].matshow(d.canvas().T, cmap='Blues')

        fft = np.fft.fftshift(np.fft.fft2(d.canvas()))
        axs[0,1].matshow(np.abs(fft), cmap='Blues')

        sigma = 1
        e = Expansion((N,N))
        Gx, Gy = e.g_vectors
        islands_data = [ (transform(isl["type"], isl["params"], Gx, Gy, sigma), isl["epsilon"]) for isl in d.islands() ]
        fourier = combine_fourier_masks(islands_data, 1, inverse=False).reshape((N,N)).T
        axs[1,1].matshow(np.abs(fourier), cmap='Blues')

        real = np.fft.ifft2(np.fft.ifftshift(fourier)) * N * N
        disp = np.real(real)
        axs[1,0].matshow(disp.T, cmap='Blues', vmin=1, vmax=8)

        assert(np.mean(np.abs(d.canvas() - np.real(real))) < 0.9)

        plt.savefig('test.png')
