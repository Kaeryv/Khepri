"""
    Alternative formulation of RCWA

    Work in progress, use with caution.
"""
import numpy as np
from math import prod, floor, sqrt
from numpy.lib.scimath import sqrt as csqrt
from numpy.linalg import inv
from cmath import pi
from .constants import c
from .tools import convolution_matrix
from scipy.linalg import expm

def redheffer_product(SA, SB):
    D = SA[0,1] @ np.linalg.inv(np.eye(SA[0,0].shape[0],dtype=np.complex128) - SB[0,0] @ SA[1,1])
    F = SB[1,0] @ np.linalg.inv(np.eye(SA[0,0].shape[0],dtype=np.complex128) - SA[1,1] @ SB[0,0])
    
    S11 = SA[0, 0] + D @ SB[0, 0] @ SA[1, 0];
    S12 = D @ SB[0, 1];
    S21 = F @ SA[1, 0];
    S22 = SB[1, 1] + F @ SA[1, 1] @ SB[0, 1];
    
    S = np.array([[S11, S12], [S21, S22]])
    return S

def scattering_relection(KX, KY, W0, V0):
    I = np.eye(KX.shape[0])
    Pref = np.vstack([
        np.hstack([KX @ KY,     I - KX @ KX]),
        np.hstack([KY @ KY - I,    -KY @ KX]),
    ])
    Qref = np.vstack([
        np.hstack([KX @ KY,     I - KX @ KX]),
        np.hstack([KY @ KY - I,   - KY @ KX]),
    ])
    # Solve the eigen problem
    eigenvals, Wref = np.linalg.eig(Pref @ Qref)

    inv_lambda = np.diag(np.reciprocal(csqrt(eigenvals)))
    Vref = Qref @ Wref @ inv_lambda
    A = inv(W0) @ Wref + inv(V0) @ Vref
    B = inv(W0) @ Wref - inv(V0) @ Vref
    S11 = -inv(A) @ B
    S12 = 2 * inv(A)
    S21 = 0.5 * (A-B@inv(A) @B)
    S22 = B @ inv(A)
    return np.array([[S11, S12], [S21, S22]])

def free_space_eigenmodes(KX, KY):
    I = np.eye(KX.shape[0])
    P = np.vstack([
        np.hstack([KX @ KY, I - KX @ KX]),
        np.hstack([KY @ KY - I, -KX @ KY]),
        ])
    Q = P
    lam2, W0 = np.linalg.eig(P @ Q)
    V0 = Q @ W0 @ np.diag(1./ csqrt(lam2))
    return W0, V0

class Lattice:
    def __init__(self, pw, a,  wavelength, kp=(0,0)):
        self.pw = pw
        kx0, ky0 = kp
        self.W0 = None
        self.V0 = None
        M = (pw[0]-1)//2
        m = np.arange(-M, M+1)
        kx = kx0 - 2 * pi * m / a
        ky = ky0 - 2 * pi * m / a
        self.KX, self.KY = np.meshgrid(kx, ky)
        self.k0 = 2 * pi / wavelength
        k0 = self.k0

        epsi = 1
        self.KZ = np.conj(csqrt(k0**2*epsi-self.KX**2-self.KY**2))

        self.KX = np.diag(self.KX.flatten()) / k0
        self.KY = np.diag(self.KY.flatten()) / k0
        self.KZ = np.diag(self.KZ.flatten()) / k0

        KX = self.KX
        KY = self.KY
        # Eigen modes of free space
        self.W0, self.V0 = free_space_eigenmodes(KX, KY)

def scattering_layer(lattice, eps, depth=400e-9):
    KX = lattice.KX
    KY = lattice.KY
    C = convolution_matrix(eps, lattice.pw)
    S = build_scatmat(lattice.k0, KX, KY, C, lattice.W0, lattice.V0, depth) 
    return S

def scattering_identity(pw):
    I = np.eye(prod(pw))
    SI = np.vstack([
        np.hstack([np.zeros_like(I), I]),
        np.hstack([I, np.zeros_like(I)]),
    ])
    return SI

def incident(pw, p_pol, s_pol, kp):
    # Normalize in pol basis
    pol_norm = sqrt(p_pol**2 + s_pol**2)
    p_pol /= pol_norm
    s_pol /= pol_norm

    knorm = np.array(kp) / np.linalg.norm(kp)
    deviceNormalUnitVector = np.array([0, 0, -1], dtype=np.complex128)
    if abs(knorm[0]) < 1e-4:
        aTE = np.array([0,1,0])
    else:
        aTE = - np.cross(deviceNormalUnitVector, knorm)
    aTE = aTE / np.linalg.norm(aTE)
    aTM = np.cross(aTE, knorm)
    aTM /= np.linalg.norm(aTM)
    N = prod(pw)
    delta = np.zeros(N, dtype=np.complex128)
    delta[(N-1)//2] = 1
    pxy = s_pol * aTE + p_pol * aTM
    return np.hstack([delta*pxy[0], delta*pxy[1]])


def build_scatmat(k0, KX, KY, C, W0, V0, dlayer):
    I = np.eye(KX.shape[0])
    IC = inv(C)
    Pi = np.vstack([
        np.hstack([KX @ IC @ KY,     I - KX @ IC @ KX]),
        np.hstack([KY @ IC @ KY - I,   - KY @ IC @ KX]),
    ])
    Qi = np.vstack([
        np.hstack([KX @ KY,     C - KX @ KX]),
        np.hstack([KY @ KY - C,   - KY @ KX]),
    ])
    
    lam2i, WI = np.linalg.eig(Pi @ Qi)
    inv_lambdas = np.diag(np.reciprocal(csqrt(lam2i)))
    lambdas = np.diag(csqrt(lam2i))
    VI = Qi @ WI @ inv_lambdas
    A = inv(WI) @ W0 + inv(VI) @ V0
    B = inv(WI) @ W0 - inv(VI) @ V0
    X = expm(-k0*lambdas*dlayer)
    
    S11 = inv(A - X @ B @ inv(A) @ X @ B) @ ( X @ B @ inv(A) @ X @ A - B)
    S12 = inv(A - X @ B @ inv(A) @ X @ B) @ X @ (A - B @ inv(A) @ B)
    S21 = S12
    S22 = S11

    return np.array([[S11, S12], [S21, S22]])
