from bast.tools import joint_subspace, _joint_subspace
from bast.alternative import free_space_eigenmodes
import numpy as np
from bast.layer import Layer
from bast.expansion import Expansion

from enum import IntEnum

class Extension(IntEnum):
    NONE=0
    UP=1
    DOWN=2

class ExtendedLayer():
    def __init__(self, expansion, base_layer, gs, mode) -> None:
        self.expansion = expansion
        self.base = base_layer
        self.gs = gs
        self.depth = self.base.depth
        self.mode = mode

    def solve(self, k_parallel, wavelength):
        Ss = list()
        WIs = list()
        VIs = list()
        LIs = list()
        for kp in self.gs.T:
            self.base.solve(kp + k_parallel, wavelength)
            Ss.append(self.base.S.copy())
            WIs.append(self.base.W.copy())
            VIs.append(self.base.V.copy())
            LIs.append(np.diag(self.base.L).copy())

        mode = self.mode
        self.S =  joint_subspace(Ss, kind=mode)
        self.W = _joint_subspace(WIs, kind=mode)
        self.V = _joint_subspace(VIs, kind=mode)
        self.W0, self.V0 = extended_freespace(self.base.expansion, self.gs.T, wavelength, mode, k_parallel)
        self.L = np.diag(_joint_subspace(LIs, kind=mode))
        self.IC = 1.0


# def extended_layer(layer: Layer, expansion: Expansion, wl, gs, mode):
#     Ss = list()
#     WIs = list()
#     VIs = list()
#     LIs = list()

#     # V0s= list()
#     # W0s=list()

#     el = Layer()

#     el.expansion = expansion

#     for kp in gs.T:
#         layer.solve(kp, wl)
#         Ss.append(layer.S.copy())
#         WIs.append(layer.W.copy())
#         VIs.append(layer.V.copy())
#         LIs.append(np.diag(layer.L).copy()) # temporary, extension needs refactor
#         # Kx0, Ky0, _ = layer.expansion.k_vectors(kp, wl)
#         # W0, V0 = free_space_eigenmodes(Kx0, Ky0)
#         # W0s.append(W0)
#         # V0s.append(V0)

#     el.S =  joint_subspace(Ss, kind=mode)
#     el.W = _joint_subspace(WIs, kind=mode)
#     el.V = _joint_subspace(VIs, kind=mode)
#     el.W0, el.V0 = extended_freespace(layer.expansion, gs.T, wl, mode)
#     el.L = np.diag(_joint_subspace(LIs, kind=mode))
#     el.depth = layer.depth

#     return el

def extended_freespace(e, gs, wl, mode, k_parallel):
    V0s = list()
    W0s = list()
    for kp in gs:
        Kx0, Ky0, _ = e.k_vectors(kp + k_parallel, wl)
        W0, V0 = free_space_eigenmodes(Kx0, Ky0)
        W0s.append(W0)
        V0s.append(V0)

    W0 = _joint_subspace(W0s, kind=mode)
    V0 = _joint_subspace(V0s, kind=mode)

    return W0, V0



# def matrix_r(Ws, Vs, mode):
#     return _joint_subspace([np.block([[w, w],[-v, v]]) for w, v in zip(Ws, Vs)], mode)

# def build_extended_smat(crystal, wl, gs, mode, store_eigenspace=False):
#     Ss = list()
#     WIs = list()
#     VIs = list()
#     LIs = list()

#     for kp in gs.T:
#         crystal.set_source(wl, kp=kp)
#         crystal.solve()
#         Ss.append(crystal.Stot)
#         if store_eigenspace:
#             WIs.append(crystal.layer_eigenspace["S2"].W)
#             VIs.append(crystal.layer_eigenspace["S2"].V)
#             LIs.append(np.diag(crystal.layer_eigenspace["S2"].L))

#     if store_eigenspace:
#         return joint_subspace(Ss, kind=mode), _joint_subspace(WIs, kind=mode), _joint_subspace(VIs, kind=mode), _joint_subspace(LIs, kind=mode)
#     else:
#         return joint_subspace(Ss, kind=mode)


# class TwistedCrystal:
#     def __init__(self, c1, c2) -> None:
#         '''
#             Creates a twisted crystal from two base stacks
#         '''
#         assert c1.void and c2.void, "Twisted crystals must be expressed in void. Please use void=True when creating them."
#         self.c1 = c1
#         self.c2 = c2


#     def solve(self, twist_angle, wl):
#         '''
#             Solving the upper and lower crystals for each one's g vectors reciprocally
#         '''
#         self.wl = wl
#         self.c1.rotate_deg(0)
#         self.c2.rotate_deg(twist_angle)
#         g1s = self.c1.get_lattice(wl, (0,0)).g_vectors
#         g2s = self.c2.get_lattice(wl, (0,0)).g_vectors
#         S1 = build_extended_smat(self.c1, wl, g2s, mode=1)
#         S2, WIs, VIs, LIs = build_extended_smat(self.c2, wl, g1s, mode=0, store_eigenspace=True)
#         self.LIs = LIs
#         self.RI = None
#         self.iRI = None
#         self.lattice = self.c2.get_lattice(wl, (0,0)) + self.c1.get_lattice(wl, (0,0))
#         self.lu = None

#         self.Stot = redheffer_product(S1, S2)
#         self.S1 = S1
#         self.S2 = S2

#         W0s = _joint_subspace([self.c2.get_lattice(wl, (0,0)).W0 for i in range(9)])
#         V0s = _joint_subspace([self.c2.get_lattice(wl, (0,0)).V0 for i in range(9)])

#         self.RI = layer_eigenbasis_matrix(WIs, VIs)
#         self.R0 = layer_eigenbasis_matrix(W0s, V0s)

#     def poynting_fluxes_end(self):
#         k0 = 2 * np.pi / self.wl
#         kzi = k0
#         esrc = incident(self.lattice.pw, 1, 1, k_vector=(0,0,kzi))
#         return poynting_fluxes(self.lattice, self.Stot[1,0] @ esrc), poynting_fluxes(self.lattice, self.Stot[0,0] @ esrc)
    
#     def fields_xy(self, z, wl):
#         k0 = 2 * np.pi / self.wl
#         kzi = k0
#         c1p = incident(self.lattice.pw, 1, 1, k_vector=(0,0,kzi))
#         c1m = self.Stot[0,0] @ c1p
#         c2p = self.Stot[1,0] @ c1p
#         cdplus, cdminus = translate_mode_amplitudes2(self.S1, self.Stot, c1p, c1m,c2p)
#         d = 0.2


#         fourier_fields, self.lu = fourier_fields_from_mode_amplitudes_lu(self.RI, np.diag(self.LIs), self.R0, (cdplus, cdminus), self.lattice.k0*(d-z), lu=self.lu)
#         real_fields = [ fourier2direct(ff.reshape(self.lattice.pw), 1, target_resolution=(127,127), kp=(0,0)) for ff in fourier_fields ]


#         return real_fields[0:3], real_fields[3:]
    
#     def plot_lattice(self):
#         fig, ax = plt.subplots()
#         ax.plot(self.lattice.g_vectors[0,:], self.lattice.g_vectors[1,:], "r.")
#         fig.show()