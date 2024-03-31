import numpy as np
from scipy import special as spe
import sys

def fexpz(z: np.array):
    z = np.asarray(z)
    mask = np.abs(z) > 1e-2
    not_mask = np.logical_not(mask)
    r1 = (np.exp(z[mask]) - 1.0) / z[mask] # Classic calculation
    z1 = z[not_mask]
    r2 = 1+z1/2.*(1+z1/3.*(1+z1/4.*(1+z1/5.*(1+z1/6.*(1+z1/7)))))

    r = np.empty_like(z, dtype=complex)
    r[mask]     = r1
    r[not_mask] = r2
    return r

def transform(shape, p, Gx, Gy, sigma):
    instance = getattr(sys.modules[__name__], "transform_" + shape)
    return instance(*p, Gx, Gy, sigma)

def transform_rectangle(ll0, ll1, ur0, ur1, Gx: np.ndarray, Gy: np.ndarray, sigma):
    # 2D Fourier transform of a rectangle island
    a, b = ur0-ll0, ur1-ll1
    transform = a*b/sigma*fexpz(-1j*Gx*a)*fexpz(-1j*Gy*b)*np.exp(-1j*(ll0*Gx+ll1*Gy))
    return transform

def transform_disc(center0, center1, radius, Gx, Gy, sigma):
    # 2D Fourier transform of a cylinder island
    norm = np.sqrt(Gx*Gx + Gy*Gy) * radius
    mask = np.logical_and(np.isclose(Gx, 0.0), np.isclose(Gy,0.0))
    nmask = np.logical_not(mask)
    transform = np.zeros_like(Gx, dtype=complex)
    transform[nmask] = np.pi * radius**2 / sigma * 2 * np.exp(-1j*(center0*Gx[nmask]+center1*Gy[nmask])) * spe.jv(1.0, norm[nmask]) / norm[nmask]
    transform[mask] = np.pi * radius**2 / sigma
    return transform

def transform_uniform(Gx, Gy, sigma):
    return np.zeros_like(Gx)

def transform_pixmap(map, Gx, Gy, sigma):
    # 2D Fourier transform of any island
    transform = np.fft.fft2(map) / sigma
    return transform



from numba import njit, prange
@njit(parallel=True)
def dft(field, kx, ky, a=1):
    ffields = np.zeros_like(kx, dtype=np.complex128)
    dx = a / len(field)
    for i in prange(kx.shape[0]):
        KX = np.exp(-1j*kx[i]*dx)
        KY = np.exp(-1j*ky[i]*dx)
        KXk = 1
        KYl = 1
        for k in range(field.shape[0]):
            KYl = 1
            for l in range(field.shape[1]):
                ffields[i] += field[k,l] *  KXk * KYl #np.exp(-1j*(kx[i]*dx*k+ky[i]*dx*l))
                KYl = KYl * KY
            KXk = KXk * KX
    return ffields

@njit(parallel=True)
def idft(ffield, kx, ky, x, y):
    '''
        Turns Fourier-space fields into real-space fields using home-made DFT.
        This allows to choose when the fields should be evaluated.
        This routine is way slower for the whole unit cell.
    '''
    oshape = x.shape
    #if len(x.shape) == 1:
    #    x = x.reshape(1, 1)
    #    y = y.reshape(1, 1)
    fields = np.zeros_like(x, dtype=np.complex128)
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    dx = (xmax - xmin) / x.shape[0]
    dy = (xmax - ymin) / x.shape[1]
    for i in prange(kx.shape[0]):
        field_i = np.zeros_like(x, dtype=np.complex128)
        KX = np.exp(1j*kx[i]*dx)
        KY = np.exp(1j*ky[i]*dy)
        KXk = np.exp(1j*kx[i]*np.min(x))
        for k in range(x.shape[0]):
            KYl = np.exp(1j*ky[i]*np.min(y))
            for l in range(x.shape[1]):
                field_i[k, l] += ffield[i] *  KXk * KYl #np.exp(-1j*(kx[i]*dx*k+ky[i]*dx*l))
                KYl *= KY
            KXk *= KX
        fields += field_i
    return fields.reshape(oshape)


    coord_shape = x.shape
    x, y = x.flatten(), y.flatten()
    phase = np.exp(1j*(kx[..., np.newaxis]*x+ky[..., np.newaxis]*y))
    field = ffield[..., np.newaxis] * phase
    field = np.sum(field, axis=0)
    return field.reshape(coord_shape)