
from collections import namedtuple
from types import SimpleNamespace
from enum import IntEnum
from .tools import epsilon_g, convolution_matrix, compute_kplanar
from .fourier import transform
from .alternative import (Lattice, solve_structured_layer, build_scatmat, scattering_identity, 
    redheffer_product, incident, scattering_reflection, scattering_transmission, poynting_fluxes,
    scattering_uniform_layer)
from copy import copy
from cmath import sqrt as csqrt
import numpy as np
from bast.misc import block_split

from bast.fields import translate_mode_amplitudes2, fourier2direct2, fourier_fields_from_mode_amplitudes,fourier2direct

Island = namedtuple("Island", "shape params epsilon")

class Formulation(IntEnum):
    UNIFORM = 0
    DFT = 1
    ANALYTICAL = 2

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


    def add_layer_pixmap(self, name, epsilon, depth):
        self.layers[name] = SimpleNamespace(name=name, formulation=Formulation.DFT, epsilon=epsilon, depth=depth)

    def add_layer_uniform(self, name, epsilon_layer, depth):
        self.layers[name] = SimpleNamespace(name=name, formulation=Formulation.UNIFORM, epsilon=epsilon_layer, depth=depth)

    def add_layer_islands(self, name, epsilon_host, islands_description, depth):
        self.layers[name] = SimpleNamespace(name=name, formulation=Formulation.ANALYTICAL, fourier=None, islands=islands_description, depth=depth, epsilon_host=epsilon_host)


    def set_stacking(self, stack):
        self.stacking = copy(stack)
        self.global_stacking = ["Sref"]
        self.global_stacking.extend(stack)
        self.global_stacking.append("Strans")

    def get_lattice(self, wl, kp):
        return Lattice(self.pw, self.a, wl, kp, rotation=self.global_rotation)

    def solve(self):
        required_layers = set(self.stacking)
        lattice = Lattice(self.pw, self.a, self.source.wavelength, self.kp, rotation=self.global_rotation)
        for name in required_layers:
            current = self.layers[name]
            if current.formulation == Formulation.DFT:
                current.C = convolution_matrix(current.epsilon, lattice.pw)
                current.IC = convolution_matrix(1/current.epsilon, lattice.pw)
                current.C = current.C[lattice.trunctation][:, lattice.trunctation]
                current.IC = current.IC[lattice.trunctation][:, lattice.trunctation]
            elif current.formulation == Formulation.ANALYTICAL:
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
                

            elif current.formulation == Formulation.UNIFORM:
                S, LI, WI, VI = scattering_uniform_layer(lattice, current.epsilon, current.depth, return_eigenspace=True)
                self.layer_eigenspace[name] = SimpleNamespace(W=WI, V=VI, L=LI)
                self.layer_matrices[name] = S
                continue
            # Common part
            WI, VI, LI =  solve_structured_layer(lattice.Kx, lattice.Ky, current.C, current.IC)
            S = build_scatmat(WI, VI, lattice.W0, lattice.V0, LI, current.depth, lattice.k0)
            self.layer_eigenspace[name] = SimpleNamespace(W=WI, V=VI, L=LI)
            self.layer_matrices[name] = S


        # Now build the stack
        Kx = lattice.Kx
        Ky = lattice.Ky
        W0 = lattice.W0
        V0 = lattice.V0

        self.stack_positions.clear()
        self.stacking_matrices.clear()
        
        if not self.void:
            Sref, Wref, Vref, Lref = scattering_reflection(Kx, Ky, W0, V0)
            self.layer_eigenspace["Sref"] = SimpleNamespace(W=Wref, V=Vref, L=Lref)
            self.layers["Sref"] = SimpleNamespace(name=name, depth=0.0, epsilon_host=1.0, formulation=Formulation.UNIFORM)
            # Add incidence medium at -inf position.
            Stot = Sref.copy()
            self.stack_positions.append(-np.inf)
            self.stacking_matrices.append(Stot.copy())
        else:
            Stot = scattering_identity(self.pw, block=True)

        self.stack_positions.append(0)
        for name in self.stacking:
            Stot = redheffer_product(Stot, self.layer_matrices[name])
            self.stacking_matrices.append(Stot.copy())
            self.stack_positions.append(self.stack_positions[-1] + self.layers[name].depth)
            
        if not self.void:
            Strans, Wtrans, Vtrans, Ltrans = scattering_transmission(Kx, Ky, W0, V0)
            self.layer_eigenspace["Strans"] = SimpleNamespace(W=Wtrans, V=Vtrans, L=Ltrans)
            Stot = redheffer_product(Stot, Strans)
        
        self.Stot = Stot

        # Create the reverse stacking
        if not self.void:
            Srev = Strans.copy()
            self.stacking_reverse_matrices = [ Srev ]
        else:
            Srev = scattering_identity(self.pw, block=True)
            self.stacking_reverse_matrices = list()
        for name in reversed(self.stacking):
            Srev = redheffer_product(self.layer_matrices[name], Srev)
            self.stacking_reverse_matrices.append(Srev.copy())
        self.stacking_reverse_matrices = list(reversed(self.stacking_reverse_matrices))

        # test
        # for k in range(len(self.stacking_matrices)):
        #     Stotp = redheffer_product(self.stacking_matrices[k], self.stacking_reverse_matrices[k])
        #     print("Error", np.mean(np.abs(Stotp-Stot)))
        # exit()


    def field_slice_xy(self, z):
        # Locate the layer in which the fields are to be computed
        layer_index = np.searchsorted(self.stack_positions, z) - 1
        layer_name = self.global_stacking[layer_index]
        #print(f"The fields position z = {z} is in layer {layer_index} named {layer_name}")
        #print(f"The layer goes from {self.stack_positions[layer_index]} to {self.stack_positions[layer_index+1]}")
        if z < 0:
            zr = z
        else:
            zr = z - self.stack_positions[layer_index]
        #print(f"zr={zr}")
        LI = self.layer_eigenspace[layer_name].L
        WI = self.layer_eigenspace[layer_name].W
        VI = self.layer_eigenspace[layer_name].V

        lattice = Lattice(self.pw, self.a, self.source.wavelength, self.kp, rotation=self.global_rotation)
        Wref = self.layer_eigenspace["Sref"].W
        iWref = np.linalg.inv(Wref)
        c1p = iWref @ incident(self.pw, self.source.te, self.source.tm, k_vector=(self.kp[0], self.kp[1], self.kzi))[np.tile(lattice.trunctation,2)]
        c1m = self.Stot[0,0] @ c1p
        c2p = self.Stot[1,0] @ c1p
        cdplus, cdminus = translate_mode_amplitudes2(self.stacking_matrices[layer_index], self.stacking_reverse_matrices[layer_index], c1p, c1m,c2p)
        print(layer_name, ":", np.mean(np.abs(cdminus)), np.mean(np.abs(cdplus)), np.mean(np.abs(c1p)), np.mean(np.abs(c1m)))
        # NOTE: cdminus explodes for homogenous layer
        # But not cdplus
        # This is BEFORE propagation!
        d = self.layers[layer_name].depth
        sx, sy, ux, uy = fourier_fields_from_mode_amplitudes((WI, VI, LI), (lattice.W0, lattice.V0, None), (cdplus, cdminus), lattice.k0*(d-zr))
        
        uz = -1j * ( lattice.kx * sy -  lattice.ky * sx)
        if self.layers[layer_name].formulation == Formulation.DFT:
            sz =  -1j * self.layers[layer_name].IC @ (lattice.kx * uy - lattice.ky * ux)
        else:
            sz =  -1j * (lattice.kx * uy - lattice.ky * ux)

        #ex, ey, ez, hx, hy, hz = [ fourier2direct2(s.reshape(self.pw), lattice.kx, lattice.ky, self.a) for s in [sx, sy, sz, ux, uy, uz]]
        def s2grid(s, pw):
            grid = np.zeros(pw, dtype="complex").flatten()
            grid[lattice.trunctation] = s
            return grid.reshape(pw)

        #ex, ey, ez, hx, hy, hz = [ fourier2direct(s2grid(s, self.pw), self.a, kp=self.kp) for s in [sx, sy, sz, ux, uy, uz]]
        ex, ey, ez, hx, hy, hz = [ fourier2direct2(s2grid(s, self.pw), lattice.kx, lattice.ky, self.a) for s in [sx, sy, sz, ux, uy, uz]]
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
    

    def poynting_flux_end(self):
        lattice = Lattice(self.pw, self.a, self.source.wavelength, self.kp)
        incident_fields = incident(self.pw, self.source.te, self.source.tm, kp=(self.kp[0], self.kp[1], self.kzi))
        Wref = self.layer_eigenspace["Sref"].W
        iWref = np.linalg.inv(Wref)
        c1p =  iWref @ incident_fields[np.tile(lattice.trunctation,2)]
        Wtrans = self.layer_eigenspace["Strans"].W
        T = poynting_fluxes(lattice, Wtrans @ self.Stot[1,0] @ c1p)
        R = poynting_fluxes(lattice, Wref @ self.Stot[0,0] @ c1p)

        return R, T

