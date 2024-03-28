import numpy as np
#This version is deprecated (came from tmat)
def fourier2direct2(ffield, kx, ky, a, target_resolution=(127,127), extent=None):
    if extent is None:
        extent = (0, a, 0, a)
    from itertools import product
    x = np.linspace(extent[0], extent[1], target_resolution[0])
    y = np.linspace(extent[2], extent[3], target_resolution[1])
    ux = kx.flatten()
    uy = ky.flatten()
    ng = len(ux)
    xy = np.asarray(list(product(x, y))).reshape(*target_resolution, 2)
    uxy_x = np.outer(ux, xy[:,:,0]).reshape(ng, *target_resolution)
    uxy_y = np.outer(uy, xy[:,:,1]).reshape(ng, *target_resolution)
    fxy = np.exp(1j*(uxy_x+uxy_y))
    return np.sum(ffield.T.reshape(ng, 1, 1) * fxy, axis=0)
