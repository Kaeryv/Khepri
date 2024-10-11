import numpy as np
from scipy.special import genlaguerre
from math import prod
from khepri.fourier import dft

def rotation_matrix(polar_angle, azimuthal_angle, polarization_angle):
    cos_polar = np.cos(polar_angle)
    sin_polar = np.sin(polar_angle)
    cos_azimuthal = np.cos(azimuthal_angle)
    sin_azimuthal = np.sin(azimuthal_angle)
    cos_polarization = np.cos(polarization_angle)
    sin_polarization = np.sin(polarization_angle)

    rotation_y_matrix = np.array([
        [cos_polar, 0.0, sin_polar],
        [0.0, 1.0, 0.0],
        [-sin_polar, 0.0, cos_polar]
    ])

    rotation_z_matrix = np.array([
        [cos_azimuthal, -sin_azimuthal, 0.0],
        [sin_azimuthal, cos_azimuthal, 0.0],
        [0.0, 0.0, 1.0]
    ])

    ux = cos_azimuthal * sin_polar
    uy = sin_azimuthal * sin_polar
    uz = cos_polar
    rotation_p_matrix = np.array([
        [
            cos_polarization + ux**2 * (1 - cos_polarization),
            ux * uy * (1 - cos_polarization) - uz * sin_polarization,
            ux * uz * (1 - cos_polarization) + uy * sin_polarization
        ],
        [
            uy * ux * (1 - cos_polarization) + uz * sin_polarization,
            cos_polarization + uy**2 * (1 - cos_polarization),
            uy * uz * (1 - cos_polarization) - ux * sin_polarization
        ],
        [
            uz * ux * (1 - cos_polarization) - uy * sin_polarization,
            uz * uy * (1 - cos_polarization) + ux * sin_polarization,
            cos_polarization + uz**2 * (1 - cos_polarization)
        ]
    ])

    return rotation_p_matrix @ rotation_z_matrix @ rotation_y_matrix


def shifted_rotated_fields(
    field_fn,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    wavelength,
    beam_origin_x: np.ndarray,
    beam_origin_y: np.ndarray,
    beam_origin_z: np.ndarray,
    polar_angle: np.ndarray,
    azimuthal_angle: np.ndarray,
    polarization_angle: np.ndarray,
    **kwargs
):
    mat = rotation_matrix(polar_angle, azimuthal_angle, polarization_angle)
    mat = np.expand_dims(mat, tuple(range(x.ndim)))

    coords = np.stack([x, y, z], axis=-1).reshape(*x.shape, 3, 1)
    print(mat.shape, coords.shape)
    rotated_coords = np.linalg.solve(mat, coords)
    rotated_coords = np.split(rotated_coords, 3, axis=-2)
    xf, yf, zf = [np.squeeze(r, axis=-1) for r in rotated_coords]

    # Solve for the rotated origin.
    origin = np.stack([beam_origin_x, beam_origin_y, beam_origin_z], axis=-1).reshape(3, 1)
    origin = np.expand_dims(origin, tuple(range(0, mat.ndim - 2)))
    rotated_origin = np.linalg.solve(mat, origin)
    assert rotated_origin.size == 3
    rotated_origin = np.split(rotated_origin, 3, axis=-2)
    xf0, yf0, zf0 = [np.squeeze(r) for r in rotated_origin]

    # Compute the fields on the rotated, shifted coordinate system.
    (exr, eyr, ezr), (hxr, hyr, hzr) = field_fn(xf - xf0, yf - yf0, zf - zf0, wavelength, **kwargs)

    rotated_efield = np.stack((exr, eyr, ezr), axis=-1)
    rotated_hfield = np.stack((hxr, hyr, hzr), axis=-1)

    # Rotate the fields back onto the original coordinate system.
    efield = np.matmul(mat, rotated_efield[..., np.newaxis])
    ex, ey, ez = np.split(efield, 3, axis=-2)
    ex = np.squeeze(ex, axis=(-2, -1))
    ey = np.squeeze(ey, axis=(-2, -1))
    ez = np.squeeze(ez, axis=(-2, -1))

    hfield = np.matmul(mat, rotated_hfield[..., np.newaxis])
    hx, hy, hz = np.split(hfield, 3, axis=-2)
    hx = np.squeeze(hx, axis=(-2, -1))
    hy = np.squeeze(hy, axis=(-2, -1))
    hz = np.squeeze(hz, axis=(-2, -1))

    return np.asarray([(ex, ey, ez), (hx, hy, hz)])



def _paraxial_laguerre_gaussian_field_fn(x, y, z, wl, w0=1, er=1, p=0, l=0):
    k = 2 * np.pi / wl
    z_r = (
        np.pi * w0**2 * np.sqrt(er) / wl
    )
    N = np.abs(l) + 2 * p
    w_z = w0 * np.sqrt(1 + (z / z_r) ** 2)
    phi = np.arctan2(x, y)
    r = np.sqrt(x**2 + y**2)
    ex = (
        w0 / w_z
        * (r * np.sqrt(2) / w_z)**np.abs(l)
        * np.exp(-(r**2) / w_z**2)
        * genlaguerre(p, np.abs(l))(2*r**2/w_z**2)
        * np.exp(-1j*l*phi)
        * np.exp(
            1j
            * (
                (k * z)  # Phase
                + k * r**2 / 2 * z / (z**2 + z_r**2)  # Wavefront curvature
                - (N+1) * np.arctan(z / z_r)  # Gouy phase
            )
        )
    )
    ey = np.zeros_like(ex)
    ez = np.zeros_like(ex)
    hx = np.zeros_like(ex)
    hy = ex / np.sqrt(er)
    hz = np.zeros_like(ex)
    return (ex, ey, ez), (hx, hy, hz)

def _paraxial_gaussian_field_fn(x, y, z, wl, beam_waist=1, er=1):
    k = 2 * np.pi / wl
    z_r = (
        np.pi * beam_waist**2 * np.sqrt(er) / wl
    )
    w_z = beam_waist * np.sqrt(1 + (z / z_r) ** 2)
    r = np.sqrt(x**2 + y**2)
    ex = (
        beam_waist
        / w_z
        * np.exp(-(r**2) / w_z**2)
        * np.exp(
            1j
            * (
                (k * z)  # Phase
                + k * r**2 / 2 * z / (z**2 + z_r**2)  # Wavefront curvature
                - np.arctan(z / z_r)  # Gouy phase
            )
        )
    )
    ey = np.zeros_like(ex)
    ez = np.zeros_like(ex)
    hx = np.zeros_like(ex)
    hy = ex / np.sqrt(er)
    hz = np.zeros_like(ex)
    return (ex, ey, ez), (hx, hy, hz)
from khepri.fourier import slow_dft
from khepri.misc import split

def amplitudes_from_fields(fields, e, wl, kp, x, y, bzs, a=1):
    kxi, kyi = kp
    # Find mode coefficients
    phase = np.exp(1j*(kxi*x+kyi*y)) # There is no incidence angle.
    F = fields / phase[..., np.newaxis, np.newaxis]
    F = np.asarray(np.split(F, bzs[0], axis=1))
    F = np.asarray(np.split(F, bzs[1], axis=1))

    F = np.swapaxes(F, 0, 1)
    NS = F.shape[2]

    x = np.asarray(np.split(x, bzs[0], axis=1))
    x = np.asarray(np.split(x, bzs[1], axis=1))
    x = x.reshape(-1, NS, NS)
    y = np.asarray(np.split(y, bzs[0], axis=1))
    y = np.asarray(np.split(y, bzs[1], axis=1))
    y = y.reshape(-1, NS, NS)

    kx, ky = e.g_vectors
    F = F.reshape(-1, NS, NS, 2, 3)
    F = F[..., :2].reshape(-1, NS, NS, 4)
    Fdft = np.empty((prod(bzs), 4, prod(e.pw)), dtype=np.complex128)

    for i in range(F.shape[0]):
        for j in range(4):
            #Fdft[i, j] = dft(F[i, ..., j], kx, ky, a=a).reshape(e.pw).flatten() / F.shape[0]/F.shape[1]
            Fdft[i, j] = slow_dft(F[i, ..., j].flatten(), x[i].flatten(), y[i].flatten(), kx.flatten(), ky.flatten()) / F.shape[0]/F.shape[1]
    return np.sum(Fdft, 0)

def gen_bzi_grid(shape, a=1, reciproc=None):
    if reciproc is None:
        b1 = [2*np.pi / a, 0]
        b2 = [0, 2*np.pi / a]
    else:
        b1, b2 = reciproc
    
    si, sj = 1 / shape[0], 1 / shape[1]
    i, j = np.meshgrid(
            np.arange(-0.5 + si / 2, 0.5, si),
            np.arange(-0.5 + sj / 2, 0.5, sj),
            indexing="ij"
	    )
    return np.stack([b1[0] * i + b2[0] * j,
                     b1[1] * i + b2[1] * j])
