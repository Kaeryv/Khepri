import sys
sys.path.append(".")

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-id", type=int, required=True)
parser.add_argument("-nfreq", type=int, required=True)
parser.add_argument("-nangles", type=int, required=True)
parser.add_argument("-batch_size", type=int, default=64)
parser.add_argument("-polar", type=str, default="RCP")
args = parser.parse_args()

from bast.alternative import scattering_structured_layer, Lattice, scattering_identity, scat_base_transform,redheffer_product,incident,scattering_reflection,scattering_transmission, scattering_uniform_layer
from bast.tools import c
import numpy as np
from bast.tools import joint_subspace, rotation_matrix
from numpy.lib.scimath import sqrt as csqrt
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import product
from cmath import sqrt as csqrt
from scipy.linalg import block_diag
from numpy.linalg import inv, solve
from bast.misc import block_split


pw = (3,3)
a = 1e-2
kpinc = (0,0)

# Define the structure
N = 256
eps = 4*np.ones((N,N))
x = np.linspace(-0.5, 0.5, N, endpoint=True)
y = np.linspace(-0.5, 0.5, N, endpoint=True)
X, Y = np.meshgrid(x, y)
eps[np.sqrt(X**2+Y**2)<= 0.25] = 1.0

def get_smatrix(l, wl):
    S = scattering_structured_layer(l, eps, 0.2 * a)
    Si = scattering_uniform_layer(l, 1.0, 0.3 * a)
    S = redheffer_product(S, Si)
    S = scat_base_transform(S, l.W0)
    # Convert [2,2,N,N] to list to [2N,2N]
    S = np.asarray(np.bmat([ list(s) for s in list(S) ]))
    return S


def forall_gs(pw, a, wl, alpha, gs):
    Ss = list()
    for kp in gs.T:
        l = Lattice(pw, a, wl, kp, rotation=alpha)
        Ss.append(get_smatrix(l, wl))
    return Ss



def transmission(sl, S21, wl):
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
    else:
        assert(False)

    etm  = S21 @ esrc
    tx, ty = np.split(etm, 2)
    tz =   - (sl.Kx @ tx + sl.Ky @ ty) / np.diag(sl.Kz)

    t = sl.Kz.real @ (np.abs(tx)**2+np.abs(ty)**2+np.abs(tz)**2)
    Ttot = np.sum(t)
    return Ttot #np.sum(np.diag(sl.KZ.real) * (np.abs(tx)**2 +np.abs(ty)**2 + np.abs(tz)**2) )
def reflexion(sl, S11, wl):
    epsi=1
    k0 = 2*np.pi/wl
    kzi = np.conj(csqrt(k0**2*epsi-kpinc[0]**2-kpinc[1]**2))
    esrc = incident(sl.pw, 1, 1, kp=(sl.kp[0], sl.kp[1], kzi))
    etm  = S11 @ esrc
    rx, ry = np.split(etm, 2)
    rz = (sl.Kx @ rx + sl.Ky @ ry) / np.diag(sl.Kz)
    r = np.diag(sl.Kz.real) * (np.abs(rx)**2+np.abs(ry)**2+np.abs(rz)**2)
    Rtot = np.sum(r)
    return Rtot

def transmission_tot(alpha, f):
    wl = c / f
    g1s = Lattice(pw, a, wl, kpinc, rotation=0).g_vectors
    g2s = Lattice(pw, a, wl, kpinc, rotation=alpha).g_vectors
    
    S1s = forall_gs(pw, a, wl, 0, g2s)
    S2s = forall_gs(pw, a, wl, alpha, g1s)
    
    S1 = joint_subspace(S1s, 1)
    S2 = joint_subspace(S2s, 0)
    
    S1 = block_split(S1)
    S2 = block_split(S2)
    sl = Lattice(pw, a, wl, kpinc, rotation=alpha) + Lattice(pw, a, wl, kpinc, rotation=0)
    Stot = redheffer_product(S1, S2)
    #Sr, Wr = scattering_reflection(sl.KX, sl.KY, sl.W0, sl.V0)
    #St, Wt = scattering_transmission(sl.KX, sl.KY, sl.W0, sl.V0)
    #Stot = redheffer_product(Stot, Sr)
    #Stot = redheffer_product(St, Stot)
    return transmission(sl, Stot[1, 0], wl),reflexion(sl, Stot[0, 0], wl)

if __name__ == '__main__':
    fs = np.linspace(0.7 * c / a, 0.84 * c/a, args.nfreq)
    angles = np.linspace(1e-2, 45, args.nangles)
    angles = np.deg2rad(angles)
    Ts, Rs = list(), list()
    awl = np.asarray(list(product(fs, angles)))
    if args.id >= 0:
        work = awl[args.id * args.batch_size:(args.id+1)*args.batch_size]
        for f, angle in work:
            T,R = transmission_tot(-angle, f)
            Rs.append(R)
            Ts.append(T)
        np.savez(f"{args.polar}_3x3/RT_{args.id}.npy", T=Ts, R=Rs)
    else:
        for f,angle in tqdm(awl):
            T, R = transmission_tot(-angle, f)
            Ts.append(T)
            Rs.append(R)
        np.savez(f"{args.polar}_3x3_RT.npz", T=Ts, R=Rs)
