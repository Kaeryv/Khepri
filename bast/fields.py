from numpy.linalg import inv
import numpy as np
from scipy.linalg import block_diag

'''
    Notice:
    Here, the convention is chosen as follows:
    left ----------------- right
    Sr @ S1 @ S2 @ ... @ SN @ St
    incidence -------- emergence
    Sl -------------]d]------ Sr
      Compute Fields ^ in (d)evice
'''

def translate_mode_amplitudes(Sld, c1p, c1m):
    '''
        Sld: scattering matrix left of the layer, including the layer.
        c1p: left-side incoming fields on the stack.
        c1m: left-side outgoing fields  "   "     "
    '''
    cdminus = inv(Sld[0,1]) @ (c1m - Sld[0,0] @ c1p)
    cdplus = Sld[1,0] @ c1p + Sld[1,1] @ cdminus
    return cdplus, cdminus


def compute_z_propagator(eigenvalues, zbar):
    arg = eigenvalues * zbar
    return np.diag(np.exp(arg)), np.diag(np.exp(-arg))


def fourier2direct(ffield, a, target_resolution=(127,127), kp=(0,0)):
    '''
        Compute the real-space field from fourier representation.
    '''
    pw = ffield.shape
    kxi, kyi = kp
    ext_fft = np.pad(ffield.reshape(pw), ((target_resolution[0]-pw[0])//2, (target_resolution[0]-pw[1])//2), constant_values=0)
    xy = np.linspace(-a/2, a/2, target_resolution[0])
    XX, YY = np.meshgrid(xy,xy)
    phase = np.exp(1j * (kxi * XX + kyi * YY)).T
    F = np.fft.ifft2(np.fft.ifftshift(ext_fft)) * phase
    return F

def fourier_fields_from_mode_amplitudes(layer_eigenspace, freespace_eigenspace, mode_amplitudes, zbar):
    assert(len(layer_eigenspace) == 3)
    assert(len(freespace_eigenspace) == 3)
    assert(len(mode_amplitudes) == 2) # c+, c-
    W0, V0, _ = freespace_eigenspace
    WI, VI, LI = layer_eigenspace
    RI = np.block([[WI, WI],[-VI,VI]])
    R0 = np.block([[W0, W0],[-V0,V0]])
    L = block_diag(*compute_z_propagator(LI, zbar))
    return np.split(RI @ L @ inv(RI) @ R0  @ np.hstack(mode_amplitudes), 4)