from numpy.linalg import inv, solve
import numpy as np
from scipy.linalg import block_diag
from scipy.linalg import lu_factor, lu_solve
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
    c = 0
    check = np.linalg.cond(Sld[0,1])
    if check > 1e+10:
        # Dampening the inverse using Marquardt-Levenberg coefficient
        #c = 0*1e-14
        print(f"[ERROR] Singular S01 cond = {check:.1E}", )
        #print(np.linalg.cond(Sld[0,1]))
        #cdminus = np.zeros_like(c1p).astype(np.complex128)
        # inv(Sl01) @ (c1m - Sl00 @ c1p)
        cdminus = solve(Sld[0,1] + c * np.eye(Sld[0,1].shape[0]), (c1m - Sld[0,0] @ c1p))

    else:
        cdminus = solve(Sld[0,1] + c * np.eye(Sld[0,1].shape[0]), (c1m - Sld[0,0] @ c1p))
    cdplus = Sld[1,0] @ c1p + Sld[1,1] @ cdminus
    return cdplus, cdminus

def translate_mode_amplitudes2(Sl, Sr, c1p, c1m, c2p):
    '''
        Sld: scattering matrix left of the layer, including the layer.
        c1p: left-side incoming fields on the stack.
        c1m: left-side outgoing fields  "   "     "
    '''
    #fmmax
    #cp = Sl[0,0] @ c1p
    #F = np.eye(len(cp)) - Sl[0,1] @ Sr[1,0]
    #cdplus = solve(F, cp)
    #cdminus = Sr[1,0] @ cdplus
    #print("COND", np.linalg.cond(Sl[0,1] @ Sr[0,0]))
    #cdplus = solve(Sl[0,1] @ Sr[0,0], c1m - Sl[0,0] @ c1p)
    #cdminus = Sr[0,0] @ cdplus
    #cdminus = solve(Sr[1,0] @ np.linalg.inv(Sr[0,0]), c2p)
    # 3
    F = np.eye(len(c1p)) - Sl[1,1] @ Sr[0,0]
    cdplus = solve(F, Sl[1,0] @ c1p)
    cdminus = Sr[0,0] @ cdplus
    #cdminus = np.zeros_like(cdplus)
    return cdplus, cdminus

def compute_z_propagator(eigenvalues, zbar):
    arg = eigenvalues * zbar
    return np.exp(arg), np.exp(-arg)


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

def fourier2direct2(ffield, kx, ky, a, target_resolution=(127,127)):
    from itertools import product
    vx = np.linspace(0, a, target_resolution[0])
    vy = np.linspace(0, a, target_resolution[1])
    ux = kx.flatten()
    uy = ky.flatten()
    ng = len(ux)
    xy = np.asarray(list(product(vx, vy))).reshape(*target_resolution, 2)
    uxy_x = np.outer(ux, xy[:,:,0]).reshape(ng, *target_resolution)
    uxy_y = np.outer(uy, xy[:,:,1]).reshape(ng, *target_resolution)
    fxy = np.exp(1j*(uxy_x+uxy_y))
    return np.sum(ffield.T.reshape(ng, 1, 1) * fxy, axis=0)

def layer_eigenbasis_matrix(WI, VI):
    '''
        Matrix whose columns are the eigenmodes of the E (upper part) and H  (lower part) fields eigenmodes.
        This matrix is used to go back and forth between the eigenmodes coefficients and fields spaces.
    '''
    return np.block([[WI, WI],[-VI,VI]])

def fourier_fields_from_mode_amplitudes(RI, LI, R0, mode_amplitudes, zbar):
    '''
        Computes the fields from the mode coefficients at specified z-location (depth).
    '''
    L = np.hstack(compute_z_propagator(LI, zbar))


    return np.split(RI @  (L * solve(RI,  R0 @ np.hstack(mode_amplitudes))), 4)

def fourier_fields_from_mode_amplitudes_lu(RI, LI, R0, mode_amplitudes, zbar, lu=None):
    '''
        Computes the fields from the mode coefficients at specified z-location (depth).
        RI @ L @ inv(RI) @ R0 @ c
        Using the LU decomposition strongly speeds up the process for different z depth.
    '''
    L = np.hstack(compute_z_propagator(LI, zbar))
    if lu is None:
        lu = lu_factor(RI)

    return np.split(RI @ (L*lu_solve(lu,  R0 @ np.hstack(mode_amplitudes))), 4), lu