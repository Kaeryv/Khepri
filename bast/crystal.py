import logging

from types import SimpleNamespace
from copy import copy
from cmath import sqrt as csqrt

import numpy as np
from scipy.linalg import lu_factor
from numpy.linalg import solve

from .tools import compute_kplanar
from .alternative import (scattering_identity, incident, poynting_fluxes, free_space_eigenmodes)
from .layer import Layer
from .expansion import Expansion

from .fields import translate_mode_amplitudes2, fourier_fields_from_mode_amplitudes,fourier2real_fft, layer_eigenbasis_matrix
from .fields import fourier2real_xy, longitudinal_fields
from .layer import stack_layers

from .extension import ExtendedLayer as EL

class Crystal():
    def __init__(self, pw, a=1, void=False, epsi=1, epse=1) -> None:
        self.layers = dict()
        self.stacking_matrices = list()
        self.stack_positions = []
        self.pw = pw
        self.a = a
        self.void = void
        self.expansion = Expansion(pw, a)
        self.epsi = epsi
        self.epse = epse

    @classmethod
    def from_expansion(cls, expansion, a=1, epsi=1, epse=1, void=False):
        obj = cls(expansion.pw, a=a, epse=epse, epsi=epsi, void=void)
        obj.expansion = expansion
        return obj
    
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

    def add_layer(self, name, layer, extended=False):
        if extended:
            self.layers[name] = EL(self.expansion, layer)
        else:
            self.layers[name] = layer

    def set_device(self, layers_stack, fields_mask=None):
        '''
            Take the stacking from the user device and pre.a.ppend the incidence and emergence media.
        '''
        self.device_stack = copy(layers_stack)
        self.global_stacking = []
        if not self.void:
            self.global_stacking.append("Sref")
        self.global_stacking.extend(layers_stack)
        if not self.void:
            self.global_stacking.append("Strans")
        if fields_mask is None:
            self.stack_retain_mask = [False]*len(self.global_stacking)
        else:
            self.stack_retain_mask = [True]
            self.stack_retain_mask.extend(fields_mask)
            self.stack_retain_mask.append(True)
            for name, enabled in zip(self.global_stacking, self.stack_retain_mask):
                self.layers[name].fields |= enabled
                if hasattr(self.layers[name], "base"):
                    self.layers[name].base.fields |= enabled

    @property
    def depth(self):
        depth = 0
        for name in self.device_stack:
            depth += self.layers[name].depth
        return depth





    def solve(self):
        # Solving the required layers
        required_layers = set(self.global_stacking)
        if "Sref" not in required_layers:
            self.layers["Sref"] = Layer.half_infinite(self.expansion, "reflexion", self.epsi)
            self.layers["Sref"].fields = True
        if "Strans" not in required_layers:
            self.layers["Strans"] = Layer.half_infinite(self.expansion, "transmission", self.epse)
            self.layers["Strans"].fields = True
        for name in required_layers:
            self.layers[name].solve(self.kp, self.source.wavelength)


        # The stack is built this way:
        # If reflexion side is on the left, the position is on the right, after the layer.
        # The S matrices vector corresponds to the matrix that starts from position
        # positions = [ -inf,  0,   d1,   d1 + d2  ]
        # Stot      = [ Sref,  S1,  S12,  S12T      ]
        # Srev      = [ S12T,  S2T,  ST,   I      ]
        stacked_layers = [ self.layers[name] for name in self.global_stacking ]
        layer_sizes = [ l.depth for l in stacked_layers ]

        self.stack_positions = list(np.cumsum(layer_sizes))
        self.stack_positions[-1] = np.inf
        if not self.void:
            self.stack_positions.insert(0, -np.inf )
        

        self.stacking_matrices, self.stacking_reverse_matrices, self.Stot = stack_layers(self.expansion.pw, stacked_layers, self.stack_retain_mask)

    
    def locate_layer(self, z):
        """Locates the layer at depth z and also computes the position relative to layer.

        Args:
            z (float): z depth

        Returns:
            tuple: The layer, the index and the relative depth.
        """
        assert(z!=np.nan)
        layer_index = np.searchsorted(self.stack_positions, z) - 1
        layer_name = self.global_stacking[layer_index]
        if z < 0:
            zr = z
        else:
            zr = z - self.stack_positions[layer_index]
        
        logging.debug(f"The fields position z = {z} is in layer {layer_index} named {layer_name}")
        logging.debug(f"The layer goes from {self.stack_positions[layer_index]} to {self.stack_positions[layer_index+1]}")
        logging.debug(f"zr={zr}")
        return self.layers[layer_name], layer_index, zr

    def _fourier_fields(self, z, incident_fields, use_lu=False):
        """Returns the fourier fields in the unit cell for a depth z.

        Args:
            z (float): z depth
            incident_fields (tuple): incident fields in Fourier space

        Returns:
            _type_: fourier fields at depth z.
        """
        layer, layer_index, zr = self.locate_layer(z)
        assert layer.fields, f"Layer at {z} did not store eigenspace."
        LI, WI, VI= layer.L, layer.W, layer.V
        RI = layer_eigenbasis_matrix(WI, VI)
        if use_lu and not hasattr(layer, "luRI"):
            layer.luRI = lu_factor(RI)

        e = self.expansion
        Kx, Ky, _ = e.k_vectors(self.kp, self.source.wavelength)
        if isinstance(layer, Layer):
            W0, V0 = free_space_eigenmodes(Kx, Ky)
        else:
            W0, V0 = layer.W0, layer.V0
        R0 = layer_eigenbasis_matrix(W0, V0)
        
        k0 = 2 * np.pi / self.source.wavelength

        Wref = self.layers["Sref"].W
        Vref = self.layers["Sref"].V
        Rref = layer_eigenbasis_matrix(Wref, Vref)
        c1p = np.split(solve(Rref,  incident_fields), 2)[0]      
        c1m = self.Stot[0,0] @ c1p
        c2p = self.Stot[1,0] @ c1p
        cdplus, cdminus = translate_mode_amplitudes2(self.stacking_matrices[layer_index], self.stacking_reverse_matrices[layer_index], c1p, c1m,c2p)
        d = layer.depth
        if not use_lu:
            sx, sy, ux, uy = fourier_fields_from_mode_amplitudes(RI, LI, R0, (cdplus, cdminus), k0*(d-zr))
        else:
            sx, sy, ux, uy = fourier_fields_from_mode_amplitudes(RI, LI, R0, (cdplus, cdminus), k0*(d-zr), luRI=layer.luRI)
        sz, uz = longitudinal_fields((sx, sy, ux, uy), Kx, Ky, layer.IC)

        return sx, sy, sz, ux, uy, uz

    def field_cell_xy(self, z, incident_fields, method="fft"):
        '''
            Returns the fields in the unit cell for a depth z.
        '''
        
        e = self.expansion
        Kx, Ky, _ = e.k_vectors(self.kp, self.source.wavelength)

        ffields = self._fourier_fields(z, incident_fields)
        k0 = 2 * np.pi / self.source.wavelength

        if method == "fft":
            fields = [ fourier2real_fft(s.reshape(self.pw), self.a, kp=self.kp) for s in ffields]
        elif method == "dft":
            x = np.linspace(0, self.a, 127)
            X, Y = np.meshgrid(x,x)
            fields = [ fourier2real_xy(s, k0*Kx, k0*Ky, X, Y) for s in ffields]
        
        return np.split(np.asarray(fields), 2, axis=0)
    
    def fields_coords_xy(self, x, y, z, incident_fields, use_lu=False, kp=None):
        """Returns the fields at specified coordinates (x,y) for a depth z.

        Args:
            x (array): x coordinates
            y (array): y coordinates (must be len(x))
            z (scalar): z depth
            incident_fields (_type_): _description_

        Returns:
            tuple: contains the E and H fields.
        """
        assert(x.shape == y.shape)
        if kp is None:
            kp = self.kp
        e = self.expansion
        Kx, Ky, _ = e.k_vectors(kp, self.source.wavelength)

        ffields = self._fourier_fields(z, incident_fields, use_lu=use_lu)
        k0 = 2 * np.pi / self.source.wavelength

        fields = [ fourier2real_xy(s, k0*Kx, k0*Ky, x, y) for s in ffields]
    
        return np.split(np.asarray(fields), 2, axis=0)
    
    def fields_volume(self, x, y, z, incident_fields=None, use_lu=False):
        if incident_fields is None:
            efield = incident(self.pw, self.source.te, self.source.tm, k_vector=(self.kp[0], self.kp[1], self.kzi))
            hfield = np.zeros_like(efield)
        fields = []
        for zi in z:
            fields.append(self.fields_coords_xy(x, y, zi, np.hstack((efield, hfield)), use_lu=use_lu))
        fields = np.asarray(fields)
        return fields[:,0, ...], fields[:,1, ...]

    
    def set_source(self, wavelength, te=1.0, tm=1.0, theta=0.0, phi=0.0, kp=None):
        eps_inc = 1.0
        if kp is not None:
            self.kp = kxi, kyi = kp
            self.source = SimpleNamespace(te=te, tm=tm, theta=theta, phi=phi, wavelength=wavelength)
        else:
            self.source = SimpleNamespace(te=te, tm=tm, theta=theta, phi=phi, wavelength=wavelength)
            self.kp = kxi, kyi = compute_kplanar(eps_inc, wavelength, self.source.theta, self.source.phi)
        
        self.k0 = 2 * np.pi / wavelength
        self.kzi = np.conj(csqrt(self.k0**2 * eps_inc-kxi**2 - kyi**2))

    def poynting_flux_end(self):
        #lattice = Lattice(self.pw, self.a, self.source.wavelength, self.kp)
        incident_fields = incident(self.pw, self.source.te, self.source.tm, k_vector=(self.kp[0], self.kp[1], self.kzi))
        Wref = self.layers["Sref"].W
        iWref = np.linalg.inv(Wref)
        c1p =  iWref @ incident_fields #[np.tile(lattice.trunctation,2)]
        Wtrans = self.layers["Strans"].W
        T = poynting_fluxes(self.expansion, Wtrans @ self.Stot[1,0] @ c1p, self.kp, self.source.wavelength)
        R = poynting_fluxes(self.expansion, Wref @ self.Stot[0,0] @ c1p, self.kp, self.source.wavelength)

        return R.real, T.real
    
    @property
    def zmax(self):
        return self.stack_positions[-2]
    

