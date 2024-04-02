import sys
sys.path.append(".")
assert(len(sys.argv) > 1)
action = sys.argv[1]

from bast.crystal import Crystal
from bast.tools import _joint_subspace, joint_subspace
from bast.draw import Drawing
import numpy as np
from bast.layer import Layer
from bast.expansion import Expansion
from bast.alternative import redheffer_product, incident, poynting_fluxes
import logging
from tqdm import tqdm

#logging.getLogger().setLevel(logging.DEBUG)

pw = (3, 3)
N = 256
canvas_size = (N,N)
pattern = Drawing(canvas_size, 4)
pattern.circle((0,0), 0.25, 1.0)
#pattern.rectangle((0,0), (1, 0.25), 1.0)


def extended_layer(layer: Layer, expansion: Expansion, wl, gs, mode):
    Ss = list()
    WIs = list()
    VIs = list()
    LIs = list()

    el = Layer()

    el.expansion = expansion

    for kp in gs.T:
        layer.solve(kp, wl)
        Ss.append(layer.S)
        WIs.append(layer.W)
        VIs.append(layer.V)
        LIs.append(np.diag(layer.L)) # temporary, extension needs refactor

    el.S =  joint_subspace(Ss, kind=mode)
    el.W = _joint_subspace(WIs, kind=mode)
    el.V = _joint_subspace(VIs, kind=mode)
    el.L = _joint_subspace(LIs, kind=mode)
    el.depth = layer.depth

    return el

import matplotlib.pyplot as plt
from copy import deepcopy


def solve(freq, angle_deg):
    # Upper untwisted layer
    e1 = Expansion(pw)
    layer_up = Layer.pixmap(e1, pattern.canvas(), 0.2)
    layer_air = Layer.uniform(e1, 1.0, 0.3)

    # Lower twisted layer at angle_deg
    e2 = Expansion(pw)
    e2.rotate(angle_deg)
    layer_down = Layer.pixmap(e2, pattern.canvas(), 0.2)
    
    # Extended basis and Smats
    etw = (e1+e2)
    l1 = extended_layer(layer_up,   etw, 1 / freq, e2.g_vectors, 1)
    li = extended_layer(layer_air,  etw, 1 / freq, e2.g_vectors, 1)
    l2 = extended_layer(layer_down, etw, 1 / freq, e1.g_vectors, 0)
    Sls = [ l1.S ]
    Stot = redheffer_product(l1.S, li.S)
    Sls.append(Stot)
    Stot = redheffer_product(Stot, l2.S)
    Sls.append(Stot)
    Srs = [ redheffer_product(li.S, l2.S), l2.S, scattering_identity(etw.pw, block=True) ]
    
    return etw, Stot, (l1, li, l2, Sls, Srs)

def solve_rt(freq, angle_deg):
    k0 = 2*np.pi*freq
    
    etw, Stot, _ = solve(freq, angle_deg)

    # Let's propagate a plane wave and get RT
    esrc = incident(etw.pw, 1, 1j, k_vector=(0, 0, k0))
   
    S21, S11 = Stot[1,0], Stot[0,0]
    return poynting_fluxes(etw, S11 @ esrc, (0,0), 1/freq), poynting_fluxes(etw, S21 @ esrc, (0,0), 1/freq)

from bast.fields import layer_eigenbasis_matrix, translate_mode_amplitudes2, fourier2real_xy
from bast.fields import fourier_fields_from_mode_amplitudes
from bast.alternative import free_space_eigenmodes, scattering_identity
from math import prod

def solve_fields(freq, angle_deg, x, y, z, solveresults, lu=None):
    k0 = 2 * np.pi * freq
    etw, Stot, (l1, li, l2, Sls, Srs) = solveresults

    # First, locate the layer
    if z < l1.depth:
        Sl = Sls[0]
        Sr = Srs[0]
        layer = l1
        #mode = 1
        zr = z - 0
    elif z < l1.depth + li.depth:
        Sl = Sls[1]
        Sr = Srs[1]
        layer = li
        #mode = 1
        zr = z - 0.2
    elif z < l1.depth + li.depth + l2.depth:
        Sl = Sls[2]
        Sr = Srs[2]
        layer = l2
        #mode = 0
        zr = z - 0.2 - 0.3
    else:
        print("Unsupported depth")
    #e0 = Expansion(pw)
    #Kx0, Ky0, _ = e0.k_vectors((0,0), 1/freq)
    #W0, V0 = free_space_eigenmodes(Kx0, Ky0)
    Kx, Ky, _ = etw.k_vectors((0,0), 1/freq)
    W0, V0 = free_space_eigenmodes(Kx, Ky)
    
    #W0 = _joint_subspace([W0]*prod(pw), kind=0) # TODO understand why mode zero is required.
    #V0 = _joint_subspace([V0]*prod(pw), kind=0)

    LI, WI, VI= layer.L, layer.W, layer.V
    LI = np.diag(LI)
    RI = layer_eigenbasis_matrix(WI, VI)
    R0 = layer_eigenbasis_matrix(W0, V0)

    c1p = incident(etw.pw, 1, 0, k_vector=(0, 0, k0))
    S21, S11 = Stot[1,0], Stot[0,0]       
    c1m = S11 @ c1p
    c2p = S21 @ c1p
    cdplus, cdminus = translate_mode_amplitudes2(Sl, Sr, c1p, c1m, c2p)
    d = layer.depth
    ffields, lu = fourier_fields_from_mode_amplitudes_lu(RI, LI, R0, (cdplus, cdminus), k0*(d-zr), lu=lu)
    
    fields = [ fourier2real_xy(s, k0*Kx, k0*Ky, x, y) for s in ffields]

    return fields

if action == "spectrum":
    freqs = np.linspace(0.7, 0.83, 100)
    assert(len(sys.argv)> 2)
    angle = float(sys.argv[2])
    spectrum = []
    for i, freq in enumerate(tqdm(freqs)):
        spectrum.append(solve_rt(freq, angle))
    plt.plot(spectrum)
    plt.savefig("debug.png")

elif action == "map":
    from bast.misc import str2linspace_args
    from itertools import product
    assert(len(sys.argv)> 3)
    angles = np.linspace(*str2linspace_args(sys.argv[2]))
    freqs = np.linspace(*str2linspace_args(sys.argv[3]))
    af = product(angles, freqs)
    spectrum = []
    for a, f in tqdm(list(af)):
        spectrum.append(solve_rt(f, a))
    np.savez_compressed(f"spectrum_{pw[0]}.npz", T=np.reshape(spectrum, (len(angles), len(freqs), 2))[:,:,1].T, angles=angles, freqs=freqs)

elif action == "plotmap":
    d = np.load(sys.argv[2])
    fig, ax = plt.subplots()
    T, angles, freqs = d["T"], d["angles"], d["freqs"]
    extent = [np.min(angles), np.max(angles), np.min(freqs), np.max(freqs)]
    #ax.plot(T[0, :], 'b-')
    im = ax.matshow(T, vmin=0, vmax=1, origin="lower", extent=extent)
    ax.set_aspect(320)
    ax.set_xlabel("Twist angle [deg]")
    ax.set_ylabel("Frequency [c/a]")
    plt.colorbar(im)
    fig.savefig("debug.png")
elif action == "fields":
    assert(len(sys.argv)> 3)
    twist_angle = float(sys.argv[2])
    frequency = float(sys.argv[3])

    x = np.linspace(0, 1.0, 64)
    y = np.linspace(0.1, 0.1, 1)
    X, Y = np.meshgrid(x, y, indexing="ij")
    print(X.shape)
    zvals = np.linspace(0, 0.2*2+0.3-0.05, 64)
    E = list()
    sr = solve(frequency, twist_angle)
    lu = None
    for i, z in enumerate(tqdm(zvals)):
        Ex, Ey, Hx, Hy = solve_fields(frequency, twist_angle, X, Y, z, solveresults=sr, lu=lu)
        E.append((Ex, Ey))
    E = np.asarray(E)
    print(E.shape)
    # E.shape = (z, c, x, y)
    Edisp = np.real(E[:, 0, :, 0])
    vmax= np.max(np.abs(Edisp))
    plt.matshow(Edisp, cmap="RdBu", origin="lower", vmin=-vmax, vmax=vmax)#extent=[ 0, np.max(x), 0, np.max(zvals)]
    #for z in [0.0, 0.2, 0.3+0.2, 0.2+0.3+0.2-0.05]:
    #    plt.axhline(z, color="w")
    plt.show()
'''
    tc.solve(twist_angle, 1/frequency)
    tc.plot_lattice()
    zs = np.linspace(0,0.2,32)
    E3D = list()
    for z in zs:
        E, H = tc.fields_xy(z, 1/frequency)
        E3D.append(E)
    Enorm = np.linalg.norm(E3D, axis=1)
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.matshow(Enorm[:, :,64], cmap="hot")
    ax2.matshow(Enorm[:, 64,:], cmap="hot")
    fig.savefig("Figures/twisted_fields.png")

'''