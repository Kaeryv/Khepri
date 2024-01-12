"""
    Alternative formulation of RCWA

    Work in progress, use with caution.

    Notations:
    S11 = S[0,0]
    S22 = S[1,1]
    S = [[S11, S12], [S21, S22]]
"""
import numpy as np
from math import prod, floor, sqrt
from numpy.lib.scimath import sqrt as csqrt
from numpy.linalg import inv
from cmath import pi
from .constants import c
from .tools import convolution_matrix
from scipy.linalg import expm
from .tools import rotation_matrix


def scat_base_transform(S, U):
    S[0,0] = U @ S[0,0] @ inv(U)
    S[0,1] = U @ S[0,1] @ inv(U)
    S[1,0] = U @ S[1,0] @ inv(U)
    S[1,1] = U @ S[1,1] @ inv(U)
    return S

def redheffer_product(SA, SB):
    I = np.eye(SA[0,0].shape[0], dtype=np.complex128)
    D = SA[0,1] @ inv(I - SB[0,0] @ SA[1,1])
    F = SB[1,0] @ inv(I - SA[1,1] @ SB[0,0])
    
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
    S21 = 0.5 * (A - B @ inv(A) @ B)
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

def kz_from_kplanar(kx, ky, k0, epsilon):
    return np.conj(csqrt(k0**2*epsilon-kx**2-ky**2))

def generate_expansion_vectors(pw, a):
    M = (pw[0] - 1) // 2
    m = np.arange(-M, M+1)
    gx = 2 * pi * m / a
    gy = 2 * pi * m / a
    gx, gy= np.meshgrid(gx, gy)
    return - gx.flatten(), - gy.flatten()

class Lattice:
    def __init__(self, pw, a,  wavelength, kp=(0,0), rotation=0, compute_eigenmodes=True):
        self.pw = pw
        self.kp = kp
        self.k0 = 2 * pi / wavelength
        self.epsi = 1

        self.gx, self.gy = generate_expansion_vectors(pw, a)

        if rotation != 0:
            R = rotation_matrix(rotation)
            self.gx, self.gy = R @ self.g_vectors

        self.kx, self.ky = kp[0] + self.gx, kp[1] + self.gy


        self.kz = kz_from_kplanar(self.kx, self.ky, self.k0, self.epsi)

        # Normalize wrt k0 (magnitude of incident k-vector) and create matrices
        self.KX = np.diag(self.kx / self.k0) 
        self.KY = np.diag(self.ky / self.k0) 
        self.KZ = np.diag(self.kz / self.k0)

        # Eigen modes of free space
        if compute_eigenmodes:
            self.W0, self.V0 = free_space_eigenmodes(self.KX, self.KY)

    @property
    def g_vectors(self):
        return np.vstack((self.gx, self.gy))

    def __add__(self, rhs):
        pw = self.pw
        g = np.zeros((pw[0]**2, pw[1]**2, 2), dtype=np.complex128)
        
        g2 = np.array(list(rhs.g_vectors.T))
        for i, g1 in enumerate(self.g_vectors.T):
            g[i, :, :] = g1.reshape(1, 2) + g2
        
        l = Lattice((self.pw[0]**2, self.pw[1]**2), 1e-7, 2*np.pi/self.k0, (0,0), 0, False)
        l.gx = g[:, :, 0].flatten()
        l.gy = g[:, :, 1].flatten()

        l.kx, l.ky = self.kp[0] - l.gx, self.kp[1] - l.gy


        l.kz = kz_from_kplanar(l.kx, l.ky, l.k0, self.epsi)

        # Normalize wrt k0 (magnitude of incident k-vector) and create matrices
        l.KX = np.diag(l.kx / l.k0) 
        l.KY = np.diag(l.ky / l.k0) 
        l.KZ = np.diag(l.kz / l.k0)

        # Eigen modes of free space
        l.W0, l.V0 = free_space_eigenmodes(l.KX, l.KY)
        return l


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
