from .tools import *
from .matrices import *
from .lattice import complex_dtype
from scipy.linalg import expm

def scattering_matrix(pw, lattice, island_type, params, island_eps, eps_host, wavelength, kp, depth=100e-9, slicing_pow=3):
    _, q = grid_size(pw)
    gx, gy = lattice.gx, lattice.gy
    boolean_field = transform(island_type, params, lattice.Gx, lattice.Gy, lattice.area)
    U = lattice.U(wavelength, kp)
    Vi =lattice.Vi(wavelength, kp)

    epsg = epsilon_g(
            q, [boolean_field], [island_eps], eps_host
    )
    epsinvg = epsilon_g(
            q, [boolean_field], [island_eps], eps_host, 
            inverse=True
    )
    nx, ny = gx.shape[0] // 2, gx.shape[1] // 2
    indices = np.array([ coords_from_index(pw, (nx,ny), i) for i in range(pw[0] * pw[1]) ])
    A = matrix_a(indices, gx, gy, epsg, epsinvg, wavelength, kx=kp[0], ky=kp[1], dtype=complex_dtype[lattice.dtype])
    slice_depth = lattice.dtype(depth) / lattice.dtype(2**slicing_pow)
    T = U @ expm(-A * slice_depth) @ Vi
    
    S = matrix_s(T)
    for _ in range(slicing_pow):
        S1 = S.copy()
        S = multS(S1, S1)
        del S1

    return S, boolean_field


def scattering_air(pw, lattice, wavelength, depth):
        ng = prod(pw)
        kzs = compute_kz(lattice.gx, lattice.gy, 1.0, wavelength).flatten()
        S = np.zeros((4*ng, 4*ng), dtype=np.complex128)
        for i, kz in enumerate(kzs):
                S[i, i] = np.exp(1j * kz * depth)
                S[ng+i, ng+i] = np.exp(1j * kz * depth)
                S[2*ng+i, 2*ng+i] = np.exp(1j * kz * depth)
                S[3*ng+i, 3*ng+i] = np.exp(1j * kz * depth)
        return S




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

