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
from scipy.linalg import block_diag
from .tools import unitcellarea


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
    N = len(KX)
    I = np.eye(KX.shape[0])
    # Pref = np.vstack([
    #     np.hstack([KX @ KY,     I - KX @ KX]),
    #     np.hstack([KY @ KY - I,    -KY @ KX]),
    # ])
    Qref = np.vstack([
        np.hstack([KX @ KY,     I - KX @ KX]),
        np.hstack([KY @ KY - I,   - KY @ KX]),
    ])
    arg = (I-KX**2-KY**2); #arg is kz^2
    arg = np.diag(arg)
    arg = arg.astype('complex')
    Kz = np.conj(csqrt(arg))
    eigenvals = np.hstack((1j*Kz, 1j*Kz))
    Wref = np.identity(2*N)
    # Solve the eigen problem
    # eigenvals, Wref = np.linalg.eig(Pref @ Qref)
    # eigenvals = csqrt(eigenvals)
    # inv_lambda = np.diag(np.reciprocal(eigenvals))
    Vref = Qref / eigenvals
    A = solve(W0, Wref) + solve(V0, Vref)
    B = solve(W0, Wref) - solve(V0, Vref)
    S11 = -solve(A, B)
    S12 = 2 * inv(A)
    S21 = 0.5 * (A - B @ solve(A, B))
    S22 = B @ inv(A)
    return np.array([[S11, S12], [S21, S22]]), Wref, Vref, eigenvals


def scattering_transmission(KX, KY, W0, V0):
    N = len(KX)
    I = np.eye(KX.shape[0])
    # Pref = np.vstack([
    #     np.hstack([KX @ KY,     I - KX @ KX]),
    #     np.hstack([KY @ KY - I,    -KY @ KX]),
    # ])
    Qref = np.vstack([
        np.hstack([KX @ KY,     I - KX @ KX]),
        np.hstack([KY @ KY - I,   - KY @ KX]),
    ])
    # Solve the eigen problem
    #eigenvals, Wref = np.linalg.eig(Pref @ Qref)
    arg = (I-KX**2-KY**2); #arg is kz^2
    arg = np.diag(arg)
    arg = arg.astype('complex')
    Kz = np.conj(csqrt(arg))
    eigenvals = np.hstack((1j*Kz, 1j*Kz))
    Wtrans = np.identity(2*N)
    #eigenvals = csqrt(eigenvals)
    #inv_lambda = np.diag(np.reciprocal(eigenvals))
    Vtrans = Qref /eigenvals
    A = solve(W0, Wtrans) + solve(V0, Vtrans)
    B = solve(W0, Wtrans) - solve(V0, Vtrans)
    S11 = B @ inv(A) 
    S12 = 0.5 * (A - B @ solve(A, B))
    S21 = 2 * inv(A)
    S22 = - solve(A, B)
    return np.array([[S11, S12], [S21, S22]]), Wtrans, Vtrans, eigenvals

def free_space_eigenmodes(KX, KY):
    '''
        Full method
    '''
    '''
    I = np.eye(KX.shape[0])
    P = np.vstack([
        np.hstack([KX @ KY, I - KX @ KX]),
        np.hstack([KY @ KY - I, -KX @ KY]),
        ])
    lam_squared, W0 = np.linalg.eig(P@P)
    lam = csqrt(lam_squared)
    mask = np.logical_or(lam.imag < 0.0, np.logical_and(np.isclose(lam.imag, 0.0), lam.real < 0.0))
    np.negative(lam, where=mask, out=lam)
    #V0 = P @ W0 / lam # P=Q
    '''
    '''
        Analytic method
    '''
    N = len(KX)
    I = np.identity(N)
    P = np.block([[KX*KY, I-KX**2], [KY**2-I, -KY*KX]])
    Q = P
    W = np.identity(2*N)
    arg = (I-KX**2-KY**2); #arg is kz^2
    arg = arg.astype('complex')
    arg = np.diag(arg)
    Kz = np.conj(csqrt(arg)); #conjugate enforces the negative sign convention (we also have to conjugate er and mur if they are complex)
    eigenvalues = np.hstack((1j*Kz, 1j*Kz))
    mask = np.logical_or(eigenvalues.imag < 0.0, np.logical_and(np.isclose(eigenvalues.imag, 0.0), eigenvalues.real < 0.0))
    np.negative(eigenvalues, where=mask, out=eigenvalues)
    V = Q / eigenvalues; #eigenvalue order is arbitrary (hard to compare with matlab
    return W,V

def kz_from_kplanar(kx, ky, k0, epsilon):
    arg = k0**2*epsilon-kx**2-ky**2
    kz = np.conj(np.sqrt(arg.astype("complex")))
    mask = np.logical_or(kz.imag < 0.0, np.logical_and(np.isclose(kz.imag, 0.0), kz.real < 0.0))
    np.negative(kz, where=mask, out=kz)
    return kz

def generate_expansion_vectors(pw, a):
    '''
        Note: multiply by reciprocal lattice basis for @hex
    '''
    M = (pw[0] - 1) // 2
    m = np.arange(-M, M+1)
    gx = 2 * pi * m / a
    gy = 2 * pi * m / a
    gx, gy = np.meshgrid(gx, gy)
    return - gx.flatten().astype(np.complex128),  - gy.flatten().astype(np.complex128)

class Lattice:
    def __init__(self, pw, a,  wavelength, kp=(0,0), rotation=0, compute_eigenmodes=True, eps_incident=1.0, truncate=False):
        self.pw = pw
        self.kp = kp
        self.k0 = 2 * pi / wavelength
        self.a = a
        self.gx, self.gy = generate_expansion_vectors(pw, a)
        self.eps_incident = eps_incident

        if rotation != 0:
            R = rotation_matrix(rotation)
            for i, (gx, gy) in enumerate(zip(self.gx, self.gy)):
                gxr, gyr = R @ [gx, gy]
                self.gx[i] = gxr
                self.gy[i] = gyr
            #self.gx, self.gy =   R @ self.g_vectors
        
        self.kx, self.ky = kp[0] + self.gx, kp[1] + self.gy

        if truncate:
            mx, my = self.gx / 2/np.pi*a, self.gy / 2/np.pi*a
            M = (pw[0] - 1) // 2
            self.trunctation = (mx**2+my**2) <= M**2
        else:
            self.trunctation = np.ones_like(self.kx, dtype=bool)

        self.kx = self.kx[self.trunctation]
        self.ky = self.ky[self.trunctation]

        self.kz = kz_from_kplanar(self.kx, self.ky, self.k0, self.eps_incident)

        # Normalize wrt k0 (magnitude of incident k-vector) and create matrices
        self.Kx = np.diag(self.kx / self.k0) 
        self.Ky = np.diag(self.ky / self.k0) 
        self.Kz = np.diag(self.kz / self.k0)

        # Eigen modes of free space
        if compute_eigenmodes:
            self.W0, self.V0 = free_space_eigenmodes(self.Kx, self.Ky)

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


        l.kz = kz_from_kplanar(l.kx, l.ky, l.k0, self.eps_incident)

        # Normalize wrt k0 (magnitude of incident k-vector) and create matrices
        l.Kx = np.diag(l.kx / l.k0) 
        l.Ky = np.diag(l.ky / l.k0) 
        l.Kz = np.diag(l.kz / l.k0)

        # Eigen modes of free space
        l.W0, l.V0 = free_space_eigenmodes(l.Kx, l.Ky)
        return l
    
    @property
    def area(self):
        return unitcellarea((self.a,0), (0, self.a))


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

def solve_uniform_layer(Kx, Ky, er, m_r = 1):
    '''
        Computes P & Q matrices for homogeneous layer.
    '''
    N = len(Kx);
    I = np.identity(N, dtype=np.complex128)
    P = (1/er) * np.block(
        [
            [ Kx * Ky,              er * m_r * I - Kx**2 ], 
            [ Ky**2 - m_r * er * I, - Ky * Kx ]
        ])
    Q = (er/m_r) * P
    W = np.identity(2*N)
    arg = (m_r*er*I-Kx**2-Ky**2)
    arg = arg.astype('complex')
    Kz = np.conj(csqrt(np.diag(arg)))
    eigenvalues = np.hstack((1j*Kz, 1j*Kz))
    mask = np.logical_or(eigenvalues.imag < 0.0, np.logical_and(np.isclose(eigenvalues.imag, 0.0), eigenvalues.real < 0.0))
    np.negative(eigenvalues, where=mask, out=eigenvalues)
    V = Q / eigenvalues

    return W, V, eigenvalues

def solve_structured_layer(KX, KY, C, IC):
    #KX = np.diag(kx)
    #KY = np.diag(ky)
    
    I = np.eye(KX.shape[0], dtype=np.longdouble)
    Pi = np.vstack([
        np.hstack([KX @ solve(C, KY),     I - KX @ solve(C, KX)]),
        np.hstack([KY @ solve(C, KY) - I,   - KY @ solve(C, KX)]),
    ]).astype(np.complex128)
    Qi = np.vstack([
        np.hstack([KX @ KY,     C - KX @ KX]),
        np.hstack([KY @ KY - C,   - KY @ KX]),
    ]).astype(np.complex128)
    
    lam2i, WI = np.linalg.eig(Pi @ Qi)
    lam = np.sqrt(lam2i+0j) # changed from np.sqrt()
    #lam = np.where(np.imag(lam) < 0, -lam, lam)
    #mask = np.logical_or(lam.imag < 0.0, np.logical_and(np.isclose(lam.imag, 0.0), lam.real < 0.0))
    #np.negative(lam, where=mask, out=lam)
    VI = Qi @ WI / lam
    return WI, VI, lam

def build_scatmat(WI, VI, W0, V0, lambdas, dbar, k0):
    t1 = solve(WI, W0)
    t2 = solve(VI, V0)
    A = t1 + t2
    B = t1 - t2
    X = np.diag(np.exp(-lambdas*dbar*k0))

    # Build the transfer matrix
    T = A - X @ B @ solve(A, X) @ B

    # Convert to scattering matrix
    S11 = solve(T , ( X @ B @ solve(A, X) @ A - B))
    S12 = solve(T , X @ (A - B @ solve(A, B)))
    S21, S22 = S12, S11
    return np.array([[S11, S12], [S21, S22]])

def scattering_uniform_layer(lattice, eps_layer, depth, return_eigenspace=False):
    WI, VI, LI =  solve_uniform_layer(lattice.Kx, lattice.Ky, eps_layer)
    S = build_scatmat(WI, VI, lattice.W0, lattice.V0, LI, depth, lattice.k0)
    if return_eigenspace:
        return S, LI, WI, VI
    else:
        return S

def scattering_structured_layer(lattice, epsilon_map, depth, return_eigenspace=False):
    C = convolution_matrix(epsilon_map, lattice.pw)
    IC = convolution_matrix(1/epsilon_map, lattice.pw)

    WI, VI, ev =  solve_structured_layer(lattice.Kx, lattice.Ky, C, IC)
    S = build_scatmat(WI, VI, lattice.W0, lattice.V0, ev, depth, lattice.k0)
    if return_eigenspace:
        return S, ev, WI, VI
    else:
        return S

def scattering_identity(pw, block=False):
    I = np.eye(2*prod(pw))
    if block:
        SI = SI = np.asarray([
            [np.zeros_like(I), I],
            [I, np.zeros_like(I)],
        ]).astype('complex')
    else:
        SI = np.vstack([
            np.hstack([np.zeros_like(I), I]),
            np.hstack([I, np.zeros_like(I)]),
        ]).astype('complex')
    return SI


def poynting_fluxes(lattice, c_output):
    epsi=1
    k0 = lattice.k0
    #kzi = np.conj(csqrt(k0**2*epsi-kpinc[0]**2-kpinc[1]**2))
    kzi = k0
    sx, sy = np.split(c_output, 2)
    kx, ky = lattice.kx, lattice.ky
    kz = lattice.kz
    sz = - (kx * sx + ky * sy) / kz
    t = kz.real/k0 @ (np.abs(sx)**2+np.abs(sy)**2+np.abs(sz)**2)
    return np.sum(t)
