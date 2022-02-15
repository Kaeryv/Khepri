from collections import namedtuple
from cmath import pi, sqrt
import numpy as np
from scipy.linalg import norm
from math import floor
from types import SimpleNamespace
from typing import Tuple, Dict
import json
import sys

from .fourier import transform

def nanometers(x):
    return x * 1e-9


''' Physical constants '''
c = 299792458.0
mu0 = 4 * pi * 1e-7
mu0c = mu0 * c
twopi = 2 * pi

def unitcellarea(a1:Tuple[float, float], a2: Tuple[float, float]):
    return abs(a1[0] * a2[1] - a1[1] * a2[0])

def reciproc(a1: Tuple[float, float], a2: Tuple[float, float]):
    coef = twopi / (a1[0] * a2[1] - a1[1] * a2[0])
    b1 = (  a2[1] * coef, -a2[0] * coef)
    b2 = ( -a1[1] * coef,  a1[0] * coef)
    return b1, b2

def gvectors(b1, b2, shape: Tuple[int, int]):
    cx, cy = ( s / 2.0 for s in shape )
    m1 = np.arange(shape[0], dtype=complex) - cx
    m2 = np.arange(shape[1], dtype=complex) - cy
    gx = np.add.outer(b1[0] * m1, b2[0] * m2)
    gy = np.add.outer(b1[1] * m1, b2[1] * m2)
    return gx, gy

def compute_kz(gx, gy, epsilon, wavelength, kp=[0.0, 0.0]):
    kx, ky = kp  
    #ky = 0.99* (twopi / wavelength)
    u = np.array([ kx + gx, ky + gy, np.zeros_like(gx) ] )
    unorm = norm(u, axis=0)
    kz = np.emath.sqrt( -np.power(unorm, 2) + epsilon * (twopi / wavelength)**2 )
    mask = np.logical_or(kz.imag < 0.0, np.logical_and(np.isclose(kz.imag, 0.0), kz.real < 0.0))
    np.negative(kz, where=mask, out=kz)
    return kz

def compute_eta(gx, gy, kp=[0.0, 0.0]):
    kx, ky = kp
    u = np.array([ kx + gx, ky + gy, np.zeros_like(gx) ], dtype=complex)
    unorm = norm(u, axis=0)
    mask = unorm**2 < 1e-10
    not_mask = np.logical_not(mask)
    u[:, not_mask] = np.divide(u[:,not_mask], unorm[not_mask])
    u[:, mask] = np.array([[1,0,0]], dtype=complex).T
    eta = np.cross(u, [0,0,1], axis=0)
    return eta[0], eta[1]


def compute_mu(gx, gy, kzg, epsilon, wavelength, kp=[0.0, 0.0]):
    kx, ky = kp
    u = np.array([ kx + gx, ky + gy, np.zeros_like(gx) ], dtype=complex)
    unorm = norm(u, axis=0)
    mask = unorm**2 < 1e-10
    not_mask = np.logical_not(mask)
    u[:, not_mask] = np.divide(u[:, not_mask], unorm[not_mask])
    u[:, mask] = np.array([[1,0,0]]).T
    mu = kzg / (sqrt(epsilon) * twopi  / wavelength) * u
    return mu[0], mu[1]

def coords_from_index(p: Tuple[int, int], n: Tuple[int, int], g: int):
    coord = np.unravel_index(g, p, order='F')
    return np.subtract(coord, n)

def epsilon_g(q, boolean_tf, epsilon_islands, eps_host, inverse=False):
    n = q[0] // 2, q[1] // 2
    eps_g = np.zeros(q, dtype=complex)
    trsf = (lambda x: x, lambda x: 1/x) [inverse]

    eps_g[n[0], n[1]] = trsf(eps_host)
    for bf, island_eps in zip(boolean_tf, epsilon_islands):
        coef = trsf(island_eps) - trsf(eps_host)
        eps_g += coef * bf

    return eps_g

def incident(pw, E0=1.0, p_pol=0.5, s_pol=0.5):
    n = tuple(p // 2 for p in pw)
    nx, ny = n
    g0 = 0 + nx + pw[0] * (0 + ny);
    ng = pw[0] * pw[1]
    Nplus     = np.zeros(ng);
    Nplus[g0] = s_pol*E0;
    Xplus     = np.zeros(ng);
    Xplus[g0] = p_pol*E0;
    Nminus    = np.zeros(ng);
    Xminus    = np.zeros(ng);

    return np.asarray([ Nplus, Xplus, Nminus, Xminus ]).flatten()

def compute_currents(P_in, P_sca, lattice, wavelength, theta_deg, phi_deg):
    q = lattice.area / (2.0 * mu0c * twopi / wavelength)
    ng = P_in.shape[0] // 4

    j1plus  = np.zeros(ng, dtype=complex)
    j1minus = np.zeros(ng, dtype=complex)
    j3plus  = np.zeros(ng, dtype=complex)

    kgz_inc = lattice.kzi(wavelength, theta_deg, phi_deg).T.flat
    kgz_emerg = lattice.kze(wavelength, theta_deg, phi_deg).T.flat
    mask = np.abs(np.imag(kgz_inc)) <= sys.float_info.epsilon
    j1plus[mask] = q * kgz_inc[mask] * (np.abs(P_in[0:ng][mask])**2 + np.abs(P_in[ng:2*ng][mask])**2)
    j1minus[mask] = q * kgz_inc[mask]*(np.abs(P_sca[2*ng:3*ng][mask])**2+np.abs(P_sca[3*ng:4*ng][mask])**2)
    
    # Where substrate absorbs, transmission is zero
    if abs(np.imag(lattice.eps_emerg)) < sys.float_info.epsilon:
        mask = abs(np.imag(kgz_emerg)) < sys.float_info.epsilon
        j3plus[mask] = q * kgz_emerg[mask] * (np.abs(P_sca[0:ng][mask])**2 + np.abs(P_sca[ng:2*ng][mask])**2)
    
    return j1plus, j1minus, j3plus


def compute_kplanar(eps_inc, wavelength,theta_deg=0.0, phi_deg=0.0):
    phi = np.deg2rad(phi_deg)
    theta = np.deg2rad(theta_deg)
    kp = np.array([np.cos(phi), np.sin(phi)], dtype=complex)
    return kp * sqrt(eps_inc) * 2 * np.pi / wavelength * np.sin(theta)
