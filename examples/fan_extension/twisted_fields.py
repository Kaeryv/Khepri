import sys
sys.path.append(".")

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-id", type=int, required=True)
parser.add_argument("-freq", type=float, required=True)
parser.add_argument("-angle", type=float, required=True)
parser.add_argument("-polar", type=str, default="RCP")
args = parser.parse_args()

from bast.alternative import scattering_structured_layer, Lattice, scat_base_transform,redheffer_product,incident,scattering_reflection,scattering_transmission, scattering_uniform_layer, poynting_fluxes
from bast.tools import c
import numpy as np
from bast.tools import joint_subspace
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import product
from bast.misc import block_split
from bast.fields import translate_mode_amplitudes, fourier_fields_from_mode_amplitudes, fourier2direct,fourier2direct2


pw = (5,5)
a = 1
kpinc = (0,0)

# Define the structure
N = 256
eps = 4*np.ones((N,N))
x = np.linspace(-0.5, 0.5, N, endpoint=True)
y = np.linspace(-0.5, 0.5, N, endpoint=True)
X, Y = np.meshgrid(x, y)
eps[np.sqrt(X**2+Y**2)<= 0.25] = 1.0

def get_layer_smatrix(l):
    S, WL, VL, LL = scattering_structured_layer(l, eps, 0.2 * a, return_eigenspace=True)
    #Si = scattering_uniform_layer(l, 1.0, 0.3 * a)
    #S = redheffer_product(S, Si)
    S = np.asarray(np.bmat([ list(s) for s in list(S) ]))
    return S, WL, VL, LL, l.W0, l.V0
def get_interlayer_smatrix(l):
    S, WL, VL, LL = scattering_uniform_layer(l, 1.0, 0.3 * a, return_eigenspace=True)
    S = np.asarray(np.bmat([ list(s) for s in list(S) ]))
    return S, WL, VL, LL, l.W0, l.V0

def matrix_r(Ws, Vs, mode):
    return block_split(joint_subspace([np.block([[w, w],[-v, v]]) for w, v in zip(Ws, Vs)], mode))

def forall_gs(pw, a, wl, alpha, gs, mode=None, interlayer=False):
    Ss = list()
    W0s = list()
    V0s = list()
    VLs = list()
    WLs = list()
    LLs = list()
    for kp in gs.T:
        l = Lattice(pw, a, wl, kp, rotation=alpha)
        if not interlayer:
            Sl, WL, VL, LL, W0, V0 = get_layer_smatrix(l)
        else:
            Sl, WL, VL, LL, W0, V0 = get_interlayer_smatrix(l)
        Ss.append(Sl)
        W0s.append(W0)
        V0s.append(V0)
        WLs.append(WL)
        VLs.append(VL)
        LLs.append(LL)
    
    R0 = matrix_r(W0s, V0s, mode)
    RL = matrix_r(WLs, VLs, mode)
    LL = matrix_r(LLs, LLs, mode)
    return block_split(joint_subspace(Ss, mode)), R0, RL, LL[0,0]



def compute_source(sl, wl):
    epsi=1
    k0 = 2*np.pi/wl
    #kzi = np.conj(csqrt(k0**2*epsi-kpinc[0]**2-kpinc[1]**2))
    kzi = k0
    if args.polar =="RCP":
        esrc = incident(sl.pw, 1, 1j, kp=(sl.kp[0], sl.kp[1], kzi))
    elif args.polar == "LCP":
        esrc = incident(sl.pw, 1, -1j, kp=(sl.kp[0], sl.kp[1], kzi))
    elif args.polar == "S":
        esrc = incident(sl.pw, 1, 0, kp=(sl.kp[0], sl.kp[1], kzi))
    elif args.polar == "P":
        esrc = incident(sl.pw, 0, 1, kp=(sl.kp[0], sl.kp[1], kzi))
    elif args.polar == "SP":
        esrc = incident(sl.pw, 1, 1, kp=(sl.kp[0], sl.kp[1], kzi))
    else:
        assert(False)
    return esrc

def compute_field_slice(alpha, f):
    wl = c / f
    print(wl)
    g1s = Lattice(pw, a, wl, kpinc, rotation=0).g_vectors
    g2s = Lattice(pw, a, wl, kpinc, rotation=alpha).g_vectors
    
    S1, R01, RL1, L1 = forall_gs(pw, a, wl, 0,     g2s, mode=1)
    Si, R0i, Ri,  Li = forall_gs(pw, a, wl, alpha, g1s, mode=1, interlayer=True)
    S2, R02, RL2, L2 = forall_gs(pw, a, wl, alpha, g1s, mode=0)
    
    sl = Lattice(pw, a, wl, kpinc, rotation=alpha) + Lattice(pw, a, wl, kpinc, rotation=0)

    S1i = redheffer_product(S1, Si)
    Stot = redheffer_product(S1i, S2)

    c1p = compute_source(sl, wl)
    # Propagate the source in the whole stack
    c1m, c2p = Stot[0,0] @ c1p, Stot[1,0] @ c1p
    # Do that again but target one specific layer (S2)
    cdplus, cdminus = translate_mode_amplitudes(S1i, c1p, c1m)
    fourier_fields = fourier_fields_from_mode_amplitudes((np.block([list(r) for r in RL2]), np.diag(L2)), np.block([list(r) for r in R02]), (cdplus, cdminus), 0.2*a*sl.k0)
    #real_fields = [ fourier2direct(ff.reshape(sl.pw), 0*a, target_resolution=(127,127), kp=(0,0)) for ff in fourier_fields ]
    real_fields = [ fourier2direct2(ff.reshape(sl.pw), sl.kx, sl.ky, 12 *a, target_resolution=(127,127), kp=(0,0)) for ff in fourier_fields ]
    fig, axs = plt.subplots(2,2)
    for i, ax in enumerate(axs.flat):
        ax.matshow(real_fields[i].real, cmap="RdBu")
    plt.show()
    #Sr, Wr = scattering_reflection(sl.KX, sl.KY, sl.W0, sl.V0)
    #St, Wt = scattering_transmission(sl.KX, sl.KY, sl.W0, sl.V0)
    #Stot = redheffer_product(Stot, Sr)
    #Stot = redheffer_product(St, Stot)
    return 0,0

if __name__ == '__main__':
    #fs = 0.803*c/a
    #angle = np.deg2rad(18)
    #fs = 0.76*c/a
    #angle = np.deg2rad(30)
    fs = args.freq*c/a
    angle = np.deg2rad(args.angle)
    T,R = compute_field_slice(-angle, fs)
