import sys
sys.path.append(".")

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-id", type=int, required=True)
parser.add_argument("-nfreqs", type=int, required=True)
parser.add_argument("-nangles", type=int, required=True)
parser.add_argument("-batch_size", type=int, default=64)
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
    return S, WL, VL, LL, l.W0
def get_interlayer_smatrix(l):
    S, WL, VL, LL = scattering_uniform_layer(l, 1.0, 0.3 * a, return_eigenspace=True)
    S = np.asarray(np.bmat([ list(s) for s in list(S) ]))
    return S, WL, VL, LL, l.W0

def forall_gs(pw, a, wl, alpha, gs, mode=None, interlayer=False):
    Ss = list()
    W0s = list()
    for kp in gs.T:
        l = Lattice(pw, a, wl, kp, rotation=alpha)
        if not interlayer:
            Sl, WL, VL, LL, W0 = get_layer_smatrix(l)
        else:
            Sl, WL, VL, LL, W0 = get_interlayer_smatrix(l)
        Ss.append(Sl)
        W0s.append(W0)
    
    bW0 = joint_subspace([np.tile(w, (2,2)) for w in W0s], mode)
    bW0 = block_split(bW0)
    return block_split(joint_subspace(Ss, mode)), bW0[0,0]



def compute_fluxes(sl, S11, S21, wl):
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

    return poynting_fluxes(sl, S21 @ esrc), poynting_fluxes(sl, S11 @ esrc)

def transmission_tot(alpha, f):
    wl = c / f
    g1s = Lattice(pw, a, wl, kpinc, rotation=0).g_vectors
    g2s = Lattice(pw, a, wl, kpinc, rotation=alpha).g_vectors
    
    S1, W10s = forall_gs(pw, a, wl, 0,     g2s, mode=1)
    Si, Wi0s = forall_gs(pw, a, wl, alpha, g1s, mode=1, interlayer=True)
    S2, W20s = forall_gs(pw, a, wl, alpha, g1s, mode=0)
    
    sl = Lattice(pw, a, wl, kpinc, rotation=alpha) + Lattice(pw, a, wl, kpinc, rotation=0)

    Stot = redheffer_product(S1, Si)
    Stot = redheffer_product(Stot, S2)
    #Sr, Wr = scattering_reflection(sl.KX, sl.KY, sl.W0, sl.V0)
    #St, Wt = scattering_transmission(sl.KX, sl.KY, sl.W0, sl.V0)
    #Stot = redheffer_product(Stot, Sr)
    #Stot = redheffer_product(St, Stot)
    return compute_fluxes(sl, Stot[0,0], Stot[1, 0], wl)

if __name__ == '__main__':
    fs = np.linspace(0.7 * c / a, 0.84 * c/a, args.nfreqs)
    #fs = np.linspace(0.9 * c / a, 2 * c/a, args.nfreqs)
    angles = np.linspace(1e-2, 90, args.nangles)
    #angles = [6.72]
    angles = np.deg2rad(angles)
    Ts, Rs = list(), list()
    awl = np.asarray(list(product(fs, angles)))
    if args.id >= 0:
        work = awl[args.id * args.batch_size:(args.id+1)*args.batch_size]
        for f, angle in work:
            T,R = transmission_tot(angle, f)
            Rs.append(R)
            Ts.append(T)
        np.savez(f"{args.polar}_{pw[0]}x{pw[1]}/RT_{args.id}.npz", T=Ts, R=Rs)
    else:
        for f,angle in tqdm(awl):
            T, R = transmission_tot(angle, f)
            Ts.append(T)
            Rs.append(R)
        np.savez(f"{args.polar}_{pw[0]}x{pw[1]}_RT.npz", T=Ts, R=Rs)
