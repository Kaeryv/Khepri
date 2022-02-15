from .tools import *
from .matrices import *
from scipy.linalg import expm

def scattering_matrix(pw, lattice, island_type, params, island_eps, eps_host, wavelength, kp, depth=100e-9, slicing_pow=3):
    nx, ny = pw[0] // 2, pw[1] // 2
    q = (4 * nx, 4 * ny)
    gx, gy = lattice.gx, lattice.gy
    boolean_field = transform(island_type, params, lattice.Gx, lattice.Gy, lattice.area)
    U, Vi = lattice.U(wavelength, kp), lattice.Vi(wavelength, kp)
    epsg = epsilon_g(
            q, [boolean_field], [island_eps], eps_host
    )
    epsinvg = epsilon_g(
            q, [boolean_field], [island_eps], eps_host, 
            inverse=True
    )
    
    A = matrix_a(gx, gy, epsg, epsinvg, wavelength, kx=kp[0], ky=kp[1])
    slice_depth = depth / float(2**slicing_pow)
    T = U @ expm(-A * slice_depth) @ Vi

    S = matrix_s(T)

    for i in range(slicing_pow):
        S1 = S.copy()
        S = multS(S1,S1)
        del S1

    return S, boolean_field


def scattering_matrix_npy(pw, lattice, island_data, island_eps, eps_host, wavelength, theta_deg=0.0, phi_deg=0.0, depth=100e-9, slicing_pow=3):
    nx, ny = pw[0] // 2, pw[1] // 2
    q = (4 * nx, 4 * ny)
    gx, gy = lattice.gx, lattice.gy
    boolean_field = island_data.copy()
    assert boolean_field.shape == lattice.Gx.shape
    kp = lattice.kp(wavelength, theta_deg, phi_deg)
    U, Vi = lattice.U(wavelength, theta_deg, phi_deg), lattice.Vi(wavelength, theta_deg, phi_deg)
    epsg = epsilon_g(
            q, [boolean_field], [island_eps], eps_host
    )
    epsinvg = epsilon_g(
            q, [boolean_field], [island_eps], eps_host, 
            inverse=True
    )
    
    A = matrix_a(gx, gy, epsg, epsinvg, wavelength, kx=kp[0], ky=kp[1])
    slice_depth = depth / float(2**slicing_pow)
    T = U @ expm(-A * slice_depth) @ Vi
    S = matrix_s(T)

    for i in range(slicing_pow):
        S1 = S.copy()
        S = multS(S1,S1)
        del S1
    return S, boolean_field

