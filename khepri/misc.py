import numpy as np
from numpy.lib.stride_tricks import as_strided

def str2linspace_args(string):
    s = string.split(":")
    assert len(s) == 3
    mn = float(s[0])
    mx = float(s[1])
    ct = int(s[2])
    return mn, mx, ct

def block_split(S):
    shape = S.shape
    m = shape[0] // 2
    one = slice(0,m)
    two = slice(m,2*m)
    # B = [np.hsplit(half, 2) for half in np.vsplit(S, 2)]
    return np.asarray([[S[one, one], S[one, two]],[S[two, one], S[two, two]]])

def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""

    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))

def coords(xmin, xmax, ymin, ymax, zmin, zmax, resolution):
    x = np.linspace(xmin, xmax, resolution[0])
    y = np.linspace(ymin, ymax, resolution[1])
    z = np.linspace(zmin, zmax, resolution[2])
    return *np.meshgrid(x, y, indexing="xy"), z



def strided_convolution(image, weight, stride):
    im_h, im_w = image.shape
    f_h, f_w = weight.shape

    out_shape = (1 + (im_h - f_h) // stride, 1 + (im_w - f_w) // stride, f_h, f_w)
    out_strides = (image.strides[0] * stride, image.strides[1] * stride, image.strides[0], image.strides[1])
    windows = as_strided(image, shape=out_shape, strides=out_strides)

    return np.tensordot(windows, weight, axes=((2, 3), (0, 1)))
def quiver(ax, X, Y, VX, VY, subsample=2, **kwargs):
    print(VX.shape, X.shape)
    kk = subsample*2+1
    K = np.ones((kk, kk)) / kk**2
    VX, VY, X, Y = [ strided_convolution(F, K, subsample) for F in (VX, VY, X, Y) ]
    return ax.quiver(X, Y, VY,VX, scale=1.5, width=0.008,**kwargs, )


def poynting_vector(E, H, axis=-1):
    # Computes poynting vector from E, H arrays where last dimension usually distinguishes the xyz projections
    return np.cross(E, np.conj(H), axis=axis)



def ensure_array(x):
    return np.array(x)