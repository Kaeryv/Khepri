from numpy.linalg import solve
import numpy as np
from scipy.linalg import lu_solve
from khepri.fourier import idft
fourier2real_xy  = idft


'''
    Notice:
    Here, the convention is chosen as follows:
    left ----------------- right
    Sr @ S1 @ S2 @ ... @ SN @ St
    incidence -------- emergence
    Sl -------------]d]------ Sr
      Compute Fields ^ in (d)evice
'''

def translate_mode_amplitudes(Sl, Sr, c1p, c1m, c2p):
    '''
        Sld: scattering matrix left of the layer, including the layer.
        c1p: left-side incoming fields on the stack.
        c1m: left-side outgoing fields
    '''
    F = np.eye(len(c1p)) - Sl[1,1] @ Sr[0,0]
    cdplus = solve(F, Sl[1,0] @ c1p)
    cdminus = Sr[0,0] @ cdplus
    return cdplus, cdminus

def compute_z_propagator(eigenvalues, zbar):
    arg = eigenvalues * zbar
    return np.exp(arg), np.exp(-arg)

def fourier2real_fft(ffield, a, target_resolution=(127,127), kp=(0,0)):
    '''
        Compute the real-space field from a truncated fourier representation.
    '''
    pw = ffield.shape
    kxi, kyi = kp
    ext_fft = np.pad(ffield.reshape(pw), ((target_resolution[0]-pw[0])//2, (target_resolution[0]-pw[1])//2), constant_values=0)
    xy = np.linspace(0, a, target_resolution[0])
    XX, YY = np.meshgrid(xy,xy)
    phase = np.exp(1j * (kxi * XX + kyi * YY))
    F = np.fft.ifft2(np.fft.ifftshift(ext_fft)) * phase
    return F



#     kx, ky = kx.flatten(), ky.flatten()
#     phase = np.exp(-1j*(kx*x[..., np.newaxis]+ky*y[..., np.newaxis]))
#     ffield = field[...,np.newaxis] * phase
#     ffield = np.sum(ffield, axis=(0,1))
#     return ffield.reshape(coord_shape)



def real2fourier_xy(field, kx, ky, x, y):
    '''
        Turns Fourier-space fields into real-space fields using home-made DFT.
        This allows to choose when the fields should be evaluated.
        This routine is way slower for the whole unit cell.
    '''
    coord_shape = kx.shape
    kx, ky = kx.flatten(), ky.flatten()
    phase = np.exp(1j*(kx*x[..., np.newaxis]+ky*y[..., np.newaxis]))
    ffield = field[...,np.newaxis] * phase
    ffield = np.sum(ffield, axis=2)
    return ffield.reshape(coord_shape)

def layer_eigenbasis_matrix(WI, VI):
    '''
        Matrix whose columns are the eigenmodes of the E (upper part) and H  (lower part) fields eigenmodes.
        This matrix is used to go back and forth between the eigenmodes coefficients and fields spaces.
    '''
    return np.block([[WI, WI],[-VI,VI]])

def fourier_fields_from_mode_amplitudes(RI, LI, R0, mode_amplitudes, zbar, luRI=None):
    '''
        Computes the fields from the mode coefficients at specified z-location (depth).
    '''
    L = np.hstack(compute_z_propagator(LI, zbar))

    if luRI is None:
        return np.split(RI @  (L * solve(RI,    R0 @ np.hstack(mode_amplitudes))), 4)
    else:
        return np.split(RI @ (L*lu_solve(luRI,  R0 @ np.hstack(mode_amplitudes))), 4)

def isnumber(x):
    return isinstance(x, (int, float, complex)) and not isinstance(x, bool)


def longitudinal_fields(transverse_fields, kx, ky, IC=1.0):
    sx, sy, ux, uy = transverse_fields
    uz = -1j * ( kx * sy -  ky * sx)
    if isnumber(IC):
        sz =  -1j * IC * (kx * uy - ky * ux)
    else:
        sz =  -1j * IC @ (kx * uy - ky * ux)

    return sz, uz
