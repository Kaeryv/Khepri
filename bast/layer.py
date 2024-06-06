from enum import IntEnum
from bast.tools import convolution_matrix, convolution_matrix_fourier
from bast.fourier import transform, combine_fourier_masks
from bast.alternative import (
    solve_structured_layer,
    solve_uniform_layer,
    build_scatmat,
    free_space_eigenmodes,
    scattering_reflection,
    scattering_transmission,
    scattering_identity,
    redheffer_product,
)
import numpy as np

from typing import Tuple


class Formulation(IntEnum):
    UNIFORM = 0
    FFT = 1
    ANALYTICAL = 2
    HALF_SPACE_INC = 3
    HALF_SPACE_TRN = 4


class Field(IntEnum):
    X = 0
    Y = 1
    Z = 2
    NORM = 3
    POYNTING = 4


def stack_layers(pw, layers, mask):
    Stot = scattering_identity(pw, block=True)
    Sls = []
    for i, layer in enumerate(layers):
        Stot = redheffer_product(Stot, layer.S)
        if mask[i]:
            Sls.append(Stot.copy())
        else:
            Sls.append(None)

    Srev = scattering_identity(pw, block=True)
    mask = list(reversed(mask[1:]))
    Srs = []
    for i, layer in enumerate(reversed(layers[1:])):
        if mask[i]:
            Srs.append(Srev.copy())
        else:
            Srs.append(None)
        Srev = redheffer_product(layer.S.copy(), Srev)
    Srs.append(Srev.copy())
    Srs = list(reversed(Srs))
    return Sls, Srs, Stot


class Layer:
    def __init__(self) -> None:
        self.formulation = None
        self.expansion = None

        self.W = None
        self.V = None
        self.L = None

        self.S = None

        self.fields = False

    @classmethod
    def pixmap(cls, expansion, pixmap, depth):
        layer = cls()
        layer.expansion = expansion
        layer.formulation = Formulation.FFT
        layer.epsilon = pixmap
        layer.depth = depth
        return layer

    @classmethod
    def uniform(cls, expansion, epsilon, depth):
        layer = cls()
        layer.expansion = expansion
        layer.formulation = Formulation.UNIFORM
        layer.epsilon = epsilon
        layer.depth = depth
        return layer

    @classmethod
    def analytical(cls, expansion, islands_description, eps_host, depth):
        layer = cls()
        layer.expansion = expansion
        layer.formulation = Formulation.ANALYTICAL
        layer.epsilon = islands_description
        layer.eps_host = eps_host
        layer.depth = depth
        return layer

    @classmethod
    def half_infinite(cls, expansion, type, epsilon):
        layer = cls()
        if type == "reflexion":
            layer.formulation = Formulation.HALF_SPACE_INC
        elif type == "transmission":
            layer.formulation = Formulation.HALF_SPACE_TRN
        else:
            print("ERROR")
        layer.expansion = expansion
        layer.depth = 0
        layer.epsilon = epsilon
        return layer

    def solve(self, k_parallel: Tuple[float, float], wavelength: float):
        """
        Obtain the eigenspace and S-matrix from layer parameters.
        parameters:
        k_parallel: incident transverse wavevector.
        wavelength: excitation wavelength
        """
        Kx, Ky, _ = self.expansion.k_vectors(k_parallel, wavelength)
        W0, V0 = free_space_eigenmodes(Kx, Ky)
        k0 = 2 * np.pi / wavelength

        if self.formulation == Formulation.FFT:
            self.C = convolution_matrix(self.epsilon, self.expansion.pw)
            self.IC = convolution_matrix(1 / self.epsilon, self.expansion.pw)
            self.W, self.V, self.L = solve_structured_layer(Kx, Ky, self.C)
            self.S = build_scatmat(self.W, self.V, W0, V0, self.L, self.depth, k0)
        if self.formulation == Formulation.ANALYTICAL:
            sigma = self.expansion.sigma
            Gx, Gy, epw = self.expansion.g_vectors_expanded(3)
            islands_data = [ (transform(isl["type"], isl["params"], Gx, Gy, sigma), isl["epsilon"]) for isl in self.epsilon ]

            
            fourier = combine_fourier_masks(islands_data, self.eps_host, inverse=False).T
            self.C = convolution_matrix_fourier(fourier.reshape(epw), self.expansion.pw)

            fourier = combine_fourier_masks(islands_data, self.eps_host, inverse=True).T
            self.IC = convolution_matrix_fourier(fourier.reshape(epw), self.expansion.pw)

            self.W, self.V, self.L = solve_structured_layer(Kx, Ky, self.C)
            self.S = build_scatmat(self.W, self.V, W0, V0, self.L, self.depth, k0)
        elif self.formulation == Formulation.UNIFORM:
            self.W, self.V, self.L = solve_uniform_layer(Kx, Ky, self.epsilon)
            self.S = build_scatmat(self.W, self.V, W0, V0, self.L, self.depth, k0)
            self.IC = 1 / self.epsilon
        elif self.formulation == Formulation.HALF_SPACE_INC:
            self.S, self.W, self.V, self.L = scattering_reflection(
                Kx, Ky, W0, V0, self.epsilon
            )
            self.IC = 1 / self.epsilon
        elif self.formulation == Formulation.HALF_SPACE_TRN:
            self.S, self.W, self.V, self.L = scattering_transmission(
                Kx, Ky, W0, V0, self.epsilon
            )
            self.IC = 1 / self.epsilon

        if not self.fields:
            self.W = None
            self.V = None
            self.L = None
            self.IC = None

        """
            l2 = Lattice((7,7), self.a, self.source.wavelength, (0,0))
            islands_data = [ 
            ( transform(isl.shape, isl.params, 
            l2.kx.reshape((7,7)), 
            l2.ky.reshape((7,7)),
            lattice.area), 
            isl.epsilon) for isl in current.islands ]
            fourier = epsilon_g((7,7), islands_data, current.epsilon_host)
            ifourier = epsilon_g((7,7), islands_data, current.epsilon_host, inverse=True)
            current.fourier = fourier
            current.C = convolution_matrix(fourier, lattice.pw, fourier=True)
            current.IC = convolution_matrix(ifourier, lattice.pw, fourier=True)
        """
