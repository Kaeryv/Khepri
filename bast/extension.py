import logging

import numpy as np
from numpy.lib.stride_tricks import as_strided
from math import prod

from bast.layer import Layer
from bast.alternative import free_space_eigenmodes

def _joint_subspace(submatrices: list, kind=0):
    """
    Join scattering matrices into larger common space.
    
    Parameters
    ----------
    submatrices: list
        Input submatrices shape=(2*N, 2*N)
    kind: int
        Merging mode
            0 -> matrices are most densely put in the extended one
            1 -> like an outer for loop
            2 -> left to implement, outer, outer loop for 3 different lattices

    Source
    ------
    Theory for Twisted Bilayer Photonic Crystal Slabs
    Beicheng Lou, Nathan Zhao, Momchil Minkov, Cheng Guo, Meir Orenstein, and Shanhui Fan
    https://doi.org/10.1103/PhysRevLett.126.136101
    """
    # N is the number of g vectors
    N = submatrices[0].shape[0] // 2
    result = np.zeros((2*N**2, 2*N**2), dtype=submatrices[0].dtype)
    ds = result.strides[-1]

    if kind == 0:
        strides = (ds*e for e in (2*N**4,N**2,2*N**3+N, 2*N**2, 1))
    elif kind == 1:
        strides = (ds*e for e in (2*N**4,N**2,2*N**2+1, 2*N**3, N))
    else:
        raise NotImplementedError("No more than two different lattices. Feel free to contribute!")
    # (BLOCKS, BLOCKS, MATRICES, INNER, INNER)
    view = as_strided(result, (2, 2, N, N, N), strides)

    # Future note: to implement 3 lattices, the matrices list goes 2D
    # (BLOCKS, BLOCKS, MATRICES, MATRICES, INNER, INNER)
    # view = as_strided(result, (2, 2, N, N, N, N), strides)

    for i, smat in enumerate(submatrices):
        view_smat = as_strided(smat, (2, 2, N, N), (ds*e for e in (2*N**2,N, 2*N, 1)))
        view[:, :, i] = view_smat
    return result


def joint_subspace(submatrices: list, kind=0):
    '''
        Wrapper of _joint_subspace that processes 4 quadrants of smatrix
    '''
    output = [[None, None], [None, None]]
    for i in range(2):
        for j in range(2):
            output[i][j] = _joint_subspace([ sm[i,j].copy() for sm in submatrices], kind=kind)
    
    return np.asarray(output)


class ExtendedLayer():
    def __init__(self, expansion, base_layer) -> None:
        if expansion.expansion_lhs == base_layer.expansion:
            self.mode = 1
            self.gs = expansion.expansion_rhs.g_vectors
        elif expansion.expansion_rhs == base_layer.expansion:
            self.mode = 0
            self.gs = expansion.expansion_lhs.g_vectors
        else:
            raise NotImplementedError(
                    "Base layer expansion should be in the extented expansion.")
        self.expansion = expansion
        self.base = base_layer
        self.depth = self.base.depth
        self.base.fields = True
        self.fields = self.base.fields
    
    def solve(self, k_parallel, wavelength):
        Ss = list()
        WIs = list()
        VIs = list()
        LIs = list()

        for kp in self.gs.T:
            self.base.solve(kp + k_parallel, wavelength)
            Ss.append(self.base.S.copy())
            if self.fields:
                WIs.append(self.base.W.copy())
                VIs.append(self.base.V.copy())
                LIs.append(np.diag(self.base.L).copy())

        mode = self.mode
        self.S =  joint_subspace(Ss, kind=mode)
        
        if self.fields:
            self.W = _joint_subspace(WIs, kind=mode)
            self.V = _joint_subspace(VIs, kind=mode)
            self.W0, self.V0 = extended_freespace(self.base.expansion, self.gs.T, wavelength, mode, k_parallel)
            self.L = np.diag(_joint_subspace(LIs, kind=mode))

        self.IC = 1.0

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

