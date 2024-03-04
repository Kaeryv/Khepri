from bast.tools import joint_subspace, _joint_subspace
from bast.misc import block_split
from bast.alternative import Lattice, redheffer_product, incident, poynting_fluxes
from bast.fields import translate_mode_amplitudes2, fourier_fields_from_mode_amplitudes, fourier2direct
import numpy as np
import matplotlib.pyplot as plt

def matrix_r(Ws, Vs, mode):
    return _joint_subspace([np.block([[w, w],[-v, v]]) for w, v in zip(Ws, Vs)], mode)

def build_extended_smat(crystal, wl, gs, mode, store_eigenspace=False):
    Ss = list()
    WIs = list()
    VIs = list()
    LIs = list()

    for kp in gs.T:
        crystal.set_source(wl, kp=kp)
        crystal.solve()
        Ss.append(crystal.Stot)
        if store_eigenspace:
            WIs.append(crystal.layer_eigenspace["S2"].W)
            VIs.append(crystal.layer_eigenspace["S2"].V)
            LIs.append(np.diag(crystal.layer_eigenspace["S2"].L))

    if store_eigenspace:
        return joint_subspace(Ss, kind=mode), _joint_subspace(WIs, kind=mode), _joint_subspace(VIs, kind=mode), _joint_subspace(LIs, kind=mode)
    else:
        return joint_subspace(Ss, kind=mode)


class TwistedCrystal:
    def __init__(self, c1, c2) -> None:
        '''
            Creates a twisted crystal from two base stacks
        '''
        assert c1.void and c2.void, "Twisted crystals must be expressed in void. Please use void=True when creating them."
        self.c1 = c1
        self.c2 = c2


    def solve(self, twist_angle, wl):
        '''
            Solving the upper and lower crystals for each one's g vectors reciprocally
        '''
        self.wl = wl
        self.c1.rotate_deg(0)
        self.c2.rotate_deg(twist_angle)
        g1s = self.c1.get_lattice(wl, (0,0)).g_vectors
        g2s = self.c2.get_lattice(wl, (0,0)).g_vectors
        S1 = build_extended_smat(self.c1, wl, g2s, mode=1)
        S2, WIs, VIs, LIs = build_extended_smat(self.c2, wl, g1s, mode=0, store_eigenspace=True)
        self.WIs = WIs
        self.VIs = VIs
        self.LIs = LIs
        self.lattice = self.c2.get_lattice(wl, (0,0)) + self.c1.get_lattice(wl, (0,0))

        self.Stot = redheffer_product(S1, S2)
        self.S1 = S1
        self.S2 = S2

    def poynting_fluxes_end(self):
        k0 = 2 * np.pi / self.wl
        kzi = k0
        esrc = incident(self.lattice.pw, 1, 1, k_vector=(0,0,kzi))
        return poynting_fluxes(self.lattice, self.Stot[1,0] @ esrc), poynting_fluxes(self.lattice, self.Stot[0,0] @ esrc)
    
    def fields_xy(self, z, wl):
        k0 = 2 * np.pi / self.wl
        kzi = k0
        c1p = incident(self.lattice.pw, 1, 1, k_vector=(0,0,kzi))
        c1m = self.Stot[0,0] @ c1p
        c2p = self.Stot[1,0] @ c1p
        cdplus, cdminus = translate_mode_amplitudes2(self.S1, self.S2, c1p, c1m,c2p)
        d = 0.2
        W0s = _joint_subspace([self.c2.get_lattice(wl, (0,0)).W0 for i in range(9)])
        V0s = _joint_subspace([self.c2.get_lattice(wl, (0,0)).V0 for i in range(9)])
        fourier_fields = fourier_fields_from_mode_amplitudes((self.WIs, self.VIs, np.diag(self.LIs)), (W0s, V0s, None), (cdplus, cdminus), self.lattice.k0*(d-z))
        real_fields = [ fourier2direct(ff.reshape(self.lattice.pw), 1, target_resolution=(127,127), kp=(0,0)) for ff in fourier_fields ]


        return real_fields[0:3], real_fields[3:]
    
    def plot_lattice(self):
        fig, ax = plt.subplots()
        ax.plot(self.lattice.g_vectors[0,:], self.lattice.g_vectors[1,:], "r.")
        fig.show()