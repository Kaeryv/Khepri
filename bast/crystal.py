import logging
from collections import namedtuple
from types import SimpleNamespace
from enum import IntEnum
from .tools import epsilon_g, convolution_matrix, compute_kplanar
from .fourier import transform
from .alternative import (scattering_identity, 
    redheffer_product, incident, poynting_fluxes,
    scattering_uniform_layer, free_space_eigenmodes)
from copy import copy
from cmath import sqrt as csqrt
import numpy as np
from bast.misc import block_split
from bast.layer import Layer, Formulation
from bast.expansion import Expansion

from bast.fields import translate_mode_amplitudes2, fourier2direct2, fourier_fields_from_mode_amplitudes,fourier2direct, layer_eigenbasis_matrix
from typing import Tuple

Island = namedtuple("Island", "shape params epsilon")



class Crystal():
    def __init__(self, pw, a=1, void=False) -> None:
        self.layers = dict()
        self.stacking = []
        self.layer_matrices = dict()
        self.layer_eigenspace = dict()
        self.stacking_matrices = list()
        self.stack_positions = []
        self.pw = pw
        self.a = a
        self.void = void
        self.global_rotation = 0
        self.expansion = Expansion(pw, a)

    def add_layer_uniform(self, name, epsilon, depth):
        '''
            Add layer without planar structuration. It will be processed analytically.
        '''
        self.layers[name] = Layer.uniform(self.expansion, epsilon, depth)
    def add_layer_pixmap(self, name, epsilon, depth):
        '''
            Add a layer from 2D ndarray that provides eps(x,y). This method will use FFT.
        '''
        self.layers[name] = Layer.pixmap(self.expansion, epsilon, depth)

    def set_stacking(self, stack):
        '''
            Take the stacking from the user device and pre.a.ppend the incidence and emergence media.
        '''
        self.stacking = copy(stack)
        self.global_stacking = ["Sref"]
        self.global_stacking.extend(stack)
        self.global_stacking.append("Strans")

    def solve(self):
        # Solving the required layers
        required_layers = set(self.global_stacking)
        #lattice = Lattice(self.pw, self.a, self.source.wavelength, self.kp, rotation=self.global_rotation)
        expansion = self.expansion
        self.layers["Sref"] = Layer.half_infinite(self.expansion, "reflexion")
        self.layers["Strans"] = Layer.half_infinite(self.expansion, "transmission")
        for name in required_layers:
            self.layers[name].solve(self.kp, self.source.wavelength)

        # The stack is built this way:
        # If reflexion side is on the left, the position is on the right, after the layer.
        # The S matrices vector corresponds to the matrix that starts from position
        # positions = [ -inf,  0,   d1,   d1 + d2  ]
        # Stot      = [ Sref,  S1,  S12,  S12T      ]
        # Srev      = [ S12T,  S2T,  ST,   I      ]

        Stot = scattering_identity(self.expansion.pw, block=True)

        self.stack_positions.clear()
        self.stacking_matrices.clear()
        current_depth = - np.inf
        self.stack_positions.append(current_depth)
        for name in self.global_stacking:
            Stot = redheffer_product(Stot, self.layers[name].S)
            self.stacking_matrices.append(Stot.copy())
            if current_depth < 0:
                current_depth=0
            current_depth += self.layers[name].depth
            self.stack_positions.append(current_depth)
        self.Stot = Stot

        self.stacking_reverse_matrices = list()
        Srev = scattering_identity(self.pw, block=True)
        for name in reversed(self.global_stacking):
            self.stacking_reverse_matrices.append(Srev.copy())
            Srev = redheffer_product(self.layers[name].S, Srev)
        self.stacking_reverse_matrices = list(reversed(self.stacking_reverse_matrices))
        return

    def field_slice_xy(self, z):
        # Locate the layer in which the fields are to be computed
        layer_index = np.searchsorted(self.stack_positions, z) - 1
        layer_name = self.global_stacking[layer_index]
        logging.debug(f"The fields position z = {z} is in layer {layer_index} named {layer_name}")
        logging.debug(f"The layer goes from {self.stack_positions[layer_index]} to {self.stack_positions[layer_index+1]}")
        if z < 0:
            zr = z
        else:
            zr = z - self.stack_positions[layer_index]
        logging.debug(f"zr={zr}")
        LI = self.layers[layer_name].L
        WI = self.layers[layer_name].W
        VI = self.layers[layer_name].V
        RI = layer_eigenbasis_matrix(WI, VI)

        e = self.expansion
        Kx, Ky, Kz = e.k_vectors((0,0), self.source.wavelength)
        W0, V0 = free_space_eigenmodes(Kx, Ky)
        R0 = layer_eigenbasis_matrix(W0, V0)
        
        k0 = 2 * np.pi / self.source.wavelength

        Wref = self.layers["Sref"].W
        iWref = np.linalg.inv(Wref)
        c1p = iWref @ incident(self.pw, self.source.te, self.source.tm, k_vector=(self.kp[0], self.kp[1], self.kzi))#[np.tile(lattice.trunctation,2)]
        c1m = self.Stot[0,0] @ c1p
        c2p = self.Stot[1,0] @ c1p
        cdplus, cdminus = translate_mode_amplitudes2(self.stacking_matrices[layer_index], self.stacking_reverse_matrices[layer_index], c1p, c1m,c2p)
        d = self.layers[layer_name].depth
        sx, sy, ux, uy = fourier_fields_from_mode_amplitudes(RI, LI, R0, (cdplus, cdminus), k0*(d-zr))

        # Obtain longitudinal fields
        uz = -1j * ( Kx * sy -  Ky * sx)
        if self.layers[layer_name].formulation == Formulation.FFT:
            sz =  -1j * self.layers[layer_name].IC @ (k0*Kx * uy - k0*Ky * ux)
        else:
            sz =  -1j * (Kx * uy - Ky * ux)

        #ex, ey, ez, hx, hy, hz = [ fourier2direct2(s.reshape(self.pw), lattice.kx, lattice.ky, self.a) for s in [sx, sy, sz, ux, uy, uz]]
        def s2grid(s, pw):
            grid = np.zeros(pw, dtype="complex").flatten()
            grid = s #[lattice.trunctation] = s
            return grid.reshape(pw)

        #ex, ey, ez, hx, hy, hz = [ fourier2direct(s2grid(s, self.pw), self.a, kp=self.kp) for s in [sx, sy, sz, ux, uy, uz]]
        ex, ey, ez, hx, hy, hz = [ fourier2direct2(s2grid(s, self.pw), k0*Kx, k0*Ky, self.a) for s in [sx, sy, sz, ux, uy, uz]]
        E = ex, ey, ez
        H = hx, hy, hz
        
        return np.asarray(E), np.asarray(H)
    
    def fields_volume(self, zvals):
        fields = []
        for z in zvals:
            fields.append(self.field_slice_xy(z))
        fields = np.asarray(fields)
        #fields = np.swapaxes(fields, 0, 1)
        #fields = np.swapaxes(fields, 1, 2)
        return fields[:,0, :, :, :], fields[:,1, :, :, :]

    
    def set_source(self, wavelength, te=1.0, tm=1.0, theta=0.0, phi=0.0, kp=None):
        eps_inc = 1.0
        if kp is not None:
            self.kp = kxi, kyi = kp
            self.source = SimpleNamespace(te=te, tm=tm, theta=theta, phi=phi, wavelength=wavelength)
        else:
            self.source = SimpleNamespace(te=te, tm=tm, theta=theta, phi=phi, wavelength=wavelength)
            self.kp = kxi, kyi = compute_kplanar(eps_inc, wavelength, self.source.theta, self.source.phi)
        
        self.k0 = 2 * np.pi / wavelength
        self.kzi = self.k0 #np.conj(csqrt(self.k0**2 * eps_inc-kxi**2 - kyi**2))

    def rotate_deg(self, alpha):
        self.global_rotation = np.deg2rad(alpha)
    

    def poynting_flux_end(self) -> Tuple[float, float]:
        #lattice = Lattice(self.pw, self.a, self.source.wavelength, self.kp)
        incident_fields = incident(self.pw, self.source.te, self.source.tm, k_vector=(self.kp[0], self.kp[1], self.kzi))
        Wref = self.layers["Sref"].W
        iWref = np.linalg.inv(Wref)
        c1p =  iWref @ incident_fields #[np.tile(lattice.trunctation,2)]
        Wtrans = self.layers["Strans"].W
        T = poynting_fluxes(self.expansion, Wtrans @ self.Stot[1,0] @ c1p, self.kp, self.source.wavelength)
        R = poynting_fluxes(self.expansion, Wref @ self.Stot[0,0] @ c1p, self.kp, self.source.wavelength)

        return R, T

