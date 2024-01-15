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
from numpy.linalg import inv, solve
from cmath import pi
from .constants import c
from .tools import convolution_matrix
from scipy.linalg import expm, eig
from .tools import rotation_matrix


def scat_base_transform(S, U):
    S[0,0] = U @ S[0,0] @ inv(U)
    S[0,1] = U @ S[0,1] @ inv(U)
    S[1,0] = U @ S[1,0] @ inv(U)
    S[1,1] = U @ S[1,1] @ inv(U)
    return S

def redheffer_product(SA, SB):
    I = np.eye(SA[0,0].shape[0], dtype=np.complex128)
    D = I - SB[0,0] @ SA[1,1]
    F = I - SA[1,1] @ SB[0,0]
    
    S11 = SA[0, 0] + SA[0,1] @ solve(D, SB[0, 0]) @ SA[1, 0];
    S12 = SA[0,1] @ solve(D, SB[0, 1])
    S21 = SB[1,0] @ solve(F, SA[1, 0])
    S22 = SB[1, 1] + SB[1,0] @ solve(F, SA[1, 1]) @ SB[0, 1];
    
    S = np.array([[S11, S12], [S21, S22]])
    return S

def scattering_reflection(KX, KY, W0, V0):
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
    return np.array([[S11, S12], [S21, S22]]), Wref


def scattering_transmission(KX, KY, W0, V0):
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
    S11 = B @ inv(A) 
    S12 = 0.5 * (A - B @ inv(A) @ B)
    S21 = 2 * inv(A)
    S22 = - inv(A) @ B
    return np.array([[S11, S12], [S21, S22]]), Wref
from scipy.linalg import block_diag
def free_space_eigenmodes(KX, KY):
    '''
        Full method
    '''
    # I = np.eye(KX.shape[0])
    # P = np.vstack([
    #     np.hstack([KX @ KY, I - KX @ KX]),
    #     np.hstack([KY @ KY - I, -KX @ KY]),
    #     ])
    # Q = P.copy()
    # lam, W0 = np.linalg.eig(P)
    # mask = np.logical_or(lam.imag < 0.0, np.logical_and(np.isclose(lam.imag, 0.0), lam.real < 0.0))
    # np.negative(lam, where=mask, out=lam)
    # V0 = P @ W0 @ np.diag(1./ lam) # P=Q

    '''
        Analytic method
    '''
    N = len(KX)
    I = np.identity(N)
    P = np.block([[KX*KY, I-KX**2], [KY**2-I, -KY*KX]])
    Q = P
    W = np.identity(2*N)
    arg = (I-KX**2-KY**2); #arg is +kz^2
    arg = arg.astype('complex');
    Kz = np.conj(np.sqrt(arg)); #conjugate enforces the negative sign convention (we also have to conjugate er and mur if they are complex)
    eigenvalues = block_diag(1j*Kz, 1j*Kz) #determining the modes of ex, ey... so it appears eigenvalue order MATTERS...
    V = Q@np.linalg.inv(eigenvalues); #eigenvalue order is arbitrary (hard to compare with matlab
    return W,V#,Kz

    #return W0, V0

def kz_from_kplanar(kx, ky, k0, epsilon):
    arg = k0**2*epsilon-kx**2-ky**2
    kz = np.conj(np.sqrt(arg.astype("complex")))
    mask = np.logical_or(kz.imag < 0.0, np.logical_and(np.isclose(kz.imag, 0.0), kz.real < 0.0))
    np.negative(kz, where=mask, out=kz)
    return kz

def generate_expansion_vectors(pw, a):
    M = (pw[0] - 1) // 2
    m = np.arange(-M, M+1)
    gx = 2 * pi * m / a
    gy = 2 * pi * m / a
    gx, gy= np.meshgrid(gx, gy)
    #return gy.flatten(),  - gx.flatten()
    return - gx.flatten(),  - gy.flatten()

class Lattice:
    def __init__(self, pw, a,  wavelength, kp=(0,0), rotation=0, compute_eigenmodes=True):
        self.pw = pw
        self.kp = kp
        self.k0 = 2 * pi / wavelength
        self.epsi = 1

        self.gx, self.gy = generate_expansion_vectors(pw, a)

        if rotation != 0:
            R = rotation_matrix(rotation)
            for i, (gx, gy) in enumerate(zip(self.gx, self.gy)):
                gxr, gyr = R @ [gx, gy]
                self.gx[i] = gxr
                self.gy[i] = gyr
            #self.gx, self.gy =   R @ self.g_vectors
        
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
        #g = np.zeros((pw[0]**2, pw[1]**2, 2), dtype=np.complex128)
        gr = list()
        #g2 = np.array(list(rhs.g_vectors.T))
        for j, g2 in enumerate(rhs.g_vectors.T):
            for i, g1 in enumerate(self.g_vectors.T):
                gr.append(g1+g2)
        gr = np.asarray(gr)
        l = Lattice((self.pw[0]**2, self.pw[1]**2), np.nan, np.nan, (0,0), 0, False)
        l.gx = gr[:, 0]
        l.gy = gr[:, 1]
        l.k0 = self.k0
        #    g[i, :, :] = g1.reshape(1, 2) + g2
        
        #l.gx = g[:, :, 0].flatten()
        #l.gy = g[:, :, 1].flatten()

        l.kx, l.ky = self.kp[0] + l.gx, self.kp[1] + l.gy


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
    ]).astype('complex')
    return SI

def incident(pw, p_pol, s_pol, kp):
    # Normalize in pol basis
    pol_norm = sqrt(np.abs(p_pol)**2 + abs(s_pol)**2)
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
    Pi = np.vstack([
        np.hstack([KX @ solve(C, KY),     I - KX @ solve(C, KX)]),
        np.hstack([KY @ solve(C, KY) - I,   - KY @ solve(C, KX)]),
    ]).astype('complex')
    Qi = np.vstack([
        np.hstack([KX @ KY,     C - KX @ KX]),
        np.hstack([KY @ KY - C,   - KY @ KX]),
    ]).astype('complex')
    
    lam2i, WI = np.linalg.eig(Pi @ Qi)
    inv_lambdas = np.diag(np.reciprocal(csqrt(lam2i)))
    lambdas = np.diag(csqrt(lam2i))
    VI = Qi @ WI @ inv_lambdas
    A = solve(WI, W0) + solve(VI, V0)
    B = solve(WI, W0) - solve(VI, V0)
    X = np.diag(np.exp(-k0*np.diag(lambdas)*dlayer))
    T = A - X @ B @ solve(A, X) @ B
    S11 = solve(T , ( X @ B @ solve(A, X) @ A - B))
    S12 = solve(T , X @ (A - B @ solve(A, B)))
    S21 = S12
    S22 = S11

    return np.array([[S11, S12], [S21, S22]])
