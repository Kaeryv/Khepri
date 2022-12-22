from .constants import mu0c, twopi, pi
from cmath import sqrt
import numpy as np
from math import prod
from scipy.linalg import solve

from numba import njit

def matrix_u(kz, eta, mu, permittivity, wavelength, dtype):
    ng = prod(kz.shape)

    hU = mu0c / sqrt(permittivity)
    mugx, mugy = mu[0].T.flat, mu[1].T.flat
    etagx, etagy = eta[0].T.flat, eta[1].T.flat
    U = np.zeros((4*ng, 4*ng), dtype=dtype)
    c1 = sqrt(permittivity) * pi / wavelength
    st = U.strides
    Uv = np.lib.stride_tricks.as_strided(U, shape=(4, 4, ng, ng), strides=(ng*st[0], ng*st[1], st[0], st[1]))

    dg = c1 / kz.T.flatten()
    dgmugy = dg * mugy
    dgmugx = dg * mugx
    dgetagx = dg * etagx
    dgetagy = dg * etagy
    np.fill_diagonal(Uv[0,0],   dgmugy)
    np.fill_diagonal(Uv[2,0],   dgmugy)
    np.fill_diagonal(Uv[0,1], - dgmugx)
    np.fill_diagonal(Uv[2,1], - dgmugx)
    np.fill_diagonal(Uv[1,0],   dgetagy)
    np.fill_diagonal(Uv[3,0], - dgetagy)
    np.fill_diagonal(Uv[1,1], - dgetagx)
    np.fill_diagonal(Uv[3,1],   dgetagx)
    np.fill_diagonal(Uv[0,2], - hU * dgetagy)
    np.fill_diagonal(Uv[2,2],   hU * dgetagy)
    np.fill_diagonal(Uv[1,2],   hU * dgmugy)
    np.fill_diagonal(Uv[3,2],   hU * dgmugy)
    np.fill_diagonal(Uv[1,3], - hU * dgmugx)
    np.fill_diagonal(Uv[3,3], - hU * dgmugx)
    np.fill_diagonal(Uv[2,3], - hU * dgetagx)
    np.fill_diagonal(Uv[0,3],   hU * dgetagx)
    return U
        
@njit
def matrix_v(eta, mu, epsilon, dtype):
    ng = eta[0].shape[0] * eta[0].shape[1]
    V = np.zeros((4*ng, 4*ng), dtype=dtype)
    hV = sqrt(epsilon) / mu0c

    mugx, mugy = mu[0].T.flat, mu[1].T.flat
    etagx, etagy = eta[0].T.flat, eta[1].T.flat

    for g in range(ng):
        V[g,g]=etagx[g]
        V[ng+g,g]=etagy[g]
        V[2*ng+g,g]=hV*mugx[g]
        V[3*ng+g,g]=hV*mugy[g]
        V[g,ng+g]=-mugx[g]
        V[ng+g,ng+g]=-mugy[g]
        V[2*ng+g,ng+g]=hV*etagx[g]
        V[3*ng+g,ng+g]=hV*etagy[g]
        V[g,2*ng+g]=etagx[g]
        V[ng+g,2*ng+g]=etagy[g]
        V[2*ng+g,2*ng+g]=-hV*mugx[g]
        V[3*ng+g,2*ng+g]=-hV*mugy[g]
        V[g,3*ng+g]=mugx[g]
        V[ng+g,3*ng+g]=mugy[g]
        V[2*ng+g,3*ng+g]=hV*etagx[g]
        V[3*ng+g,3*ng+g]=hV*etagy[g]
    return V


@njit
def matrix_a(indices, gx, gy, eps_g, epsinv_g, wavelength, kx=0, ky=0, dtype=np.complex128):
    c1 = twopi / wavelength
    nx, ny = gx.shape[0] // 2, gx.shape[1] // 2

    ieo  = 1j / mu0c * c1
    iseo = 1j * mu0c / c1
    imo  = 1j * mu0c * c1
    ismo = 1j / mu0c / c1

    ng = gx.shape[0] * gx.shape[1]
    
    A = np.zeros((4*ng, 4*ng), dtype=dtype)
    for g_j in range(ng):
        m1_j, m2_j = indices[g_j]
        i1_j=m1_j+nx
        i2_j=m2_j+ny
        ux_j=kx+gx[i1_j,i2_j]
        uy_j=ky+gy[i1_j,i2_j]
        for g_i in range(ng):
            m1_i,m2_i = indices[g_i]
            i1_i=m1_i+nx
            i2_i=m2_i+ny
            ux_i=kx+gx[i1_i,i2_i]
            uy_i=ky+gy[i1_i,i2_i]
            dm1_ij=m1_i-m1_j
            dm2_ij=m2_i-m2_j
            i1_ij=dm1_ij+2*nx
            i2_ij=dm2_ij+2*ny
            epsg=eps_g[i1_ij,i2_ij]
            epsinvg=epsinv_g[i1_ij,i2_ij]
            if g_i==g_j:
                A[2*ng+g_i,g_j]=-ismo*ux_i*uy_j
                A[3*ng+g_i,g_j]=-ismo*uy_i*uy_j
                A[2*ng+g_i,ng+g_j]=ismo*ux_i*ux_j
                A[3*ng+g_i,ng+g_j]=ismo*uy_i*ux_j
                A[ng+g_i,2*ng+g_j]=-imo
                A[g_i,3*ng+g_j]=imo
            A[3*ng+g_i,g_j]=A[3*ng+g_i,g_j]+ieo*epsg
            A[2*ng+g_i,ng+g_j]=A[2*ng+g_i,ng+g_j]-ieo*epsg
            A[g_i,2*ng+g_j]=iseo*ux_i*epsinvg*uy_j
            A[ng+g_i,2*ng+g_j]=A[ng+g_i,2*ng+g_j]+iseo*uy_i*epsinvg*uy_j
            A[g_i,3*ng+g_j]=A[g_i,3*ng+g_j]-iseo*ux_i*epsinvg*ux_j 
            A[ng+g_i,3*ng+g_j]=-iseo*uy_i*epsinvg*ux_j
    return A

def matrix_s(T):
    ng = T.shape[0] // 4
    p = np.s_[0:2*ng]
    m = np.s_[2*ng:4*ng]
    S = np.zeros_like(T)
    S[p,p] = np.linalg.inv(T[p,p])
    S[p,m] = - S[p,p] @ T[p,m]
    S[m,p] = T[m,p] @ S[p,p]
    S[m,m] = T[m,m] + T[m,p] @ S[p,m]
    return S

#def dump_matrix(M, filename="matrix.png"):
#    fig, (axreal, axcmlx) = plt.subplots(2, 1)
#    divider = make_axes_locatable(axreal)
#    caxreal = divider.append_axes('right', size='5%', pad=0.05)
#    axreal.set_title("Magnitude")
#    m1 = axreal.matshow((np.abs(M)))
#    plt.colorbar(m1, cax=caxreal)
#    m2 = axcmlx.matshow(np.imag(M))
#    divider = make_axes_locatable(axcmlx)
#    caxcmlx = divider.append_axes('right', size='5%', pad=0.05)
#    axcmlx.set_title("Argument")
#    plt.colorbar(m2, cax=caxcmlx)
#    plt.tight_layout()
#    plt.savefig(filename)

def multS(S1, S2):
    ng = S1.shape[0] // 4
    S = np.empty_like(S1)
    I0 = np.eye(2*ng, 2*ng, dtype=S1.dtype)

    p = np.s_[0:2*ng]
    m = np.s_[2*ng:4*ng]
    
    I = I0 - (S1[p,m] @ S2[m,p])
    G1 =  solve(I, S1[p,p])
    S[p,p] = S2[p,p] @ G1
    S[m,p] = S1[m,p] + S1[m,m] @ S2[m,p] @ G1
    
    G2 = solve(I, S1[p,m])

    S[p,m] = S2[p,m] + S2[p,p] @ G2 @ S2[m,m]
    S[m,m] = S1[m,m] @ (I0 + S2[m,p] @ G2) @ S2[m,m]

    return S

def polar_transform(etag, kzi, mugi, muge, eps_incid, eps_emerg, wavelength):
    U  = matrix_u(kzi, etag, mugi, eps_incid, wavelength)
    Vi = matrix_v(etag, mugi, eps_incid, wavelength)
    Ve = matrix_v(etag, muge, eps_emerg, wavelength)
    return U, Vi, Ve
