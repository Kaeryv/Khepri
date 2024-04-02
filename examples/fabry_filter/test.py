import sys
sys.path.append(".")
import numpy as np
import matplotlib.pyplot as plt
from bast.tmat.lattice import CartesianLattice
from bast.tmat.scattering import scattering_matrix, scattering_interface
from bast.tools import c, incident, compute_fluxes
from bast.alternative import scattering_structured_layer, Lattice, scat_base_transform,redheffer_product,incident as incident2,scattering_reflection,scattering_transmission, scattering_uniform_layer
from numpy.lib.scimath import sqrt as csqrt
from tqdm import tqdm


pw = (1,1)
a = 1e-7
lattice = CartesianLattice(pw, (a,0), (0,a))
wavelengths = np.linspace(300, 730, 50)
angles = np.linspace(30, 70, 5)
e_cry = 1.30**2
e_zns = 2.35**2

def get_tmat():
    refl = list()
    for angle in tqdm(angles):
        for wl in wavelengths:
            kp = lattice.kp_angle(wl*1e-9, theta_deg=angle, phi_deg=0)
            S1 = scattering_matrix(pw, lattice, 
                "uniform", [], e_cry, e_cry, wl*1e-9, kp, depth=70e-9, slicing_pow=1)
            S2 = scattering_matrix(pw, lattice, 
                "uniform", [], e_zns, e_zns, wl*1e-9, kp, depth=120e-9, slicing_pow=1)
            
            S = S1 @ S2  @ S1 @ S2 @ S1 @ S2 @ S1
        
            Si = scattering_interface(lattice, wl, kp=kp)
            S = S @ Si
        
            pin = incident(pw, p_pol=1, s_pol=0)
            pout = S @ pin
            R,T,A =  compute_fluxes(lattice, wl, pin, pout)
            refl.append(T)
    
    return np.asarray(refl).reshape(len(angles), len(wavelengths))

def get_liu():
    refl = list()
    for angle in tqdm(angles):
        for wl in wavelengths:
            k0 = 2*np.pi/(wl*1e-9)
            kpinc = (k0 * np.sin(np.deg2rad(angle)), 0)
            l = Lattice(pw, a, wl*1e-9, kpinc)
            
            S1 = scattering_uniform_layer(l, e_cry, 70e-9)
            S2 = scattering_uniform_layer(l, e_zns, 120e-9)
                         
            S = redheffer_product(S1, S2)
            S = redheffer_product(S, S1)
            S = redheffer_product(S, S2)
            S = redheffer_product(S, S1)
            S = redheffer_product(S, S2)
            S = redheffer_product(S, S1)
            Sr, Wref = scattering_reflection(l.Kx, l.Ky, l.W0, l.V0)
            St, Wtrans = scattering_transmission(l.Kx, l.Ky, l.W0, l.V0)

            S = redheffer_product(Sr, S)
            S = redheffer_product(S, St)

            epsi=1
            kzi = np.conj(csqrt(k0**2*epsi-kpinc[0]**2-kpinc[1]**2))
            esrc = incident2(l.pw, 1, 0, kp=(l.kp[0], l.kp[1], kzi))
            e_transmitted  = S[1, 0] @ np.linalg.inv(Wref) @ esrc
            tx, ty = np.split(Wtrans @ e_transmitted, 2)
            tz = - (l.Kx @ tx + l.Ky @ ty) / np.diag(l.Kz)
            t = np.diag(l.Kz.real/kzi.real*k0.real) * (np.abs(tx)**2+np.abs(ty)**2+np.abs(tz)**2)
            R = np.sum(t)
            refl.append(R)
    
    return np.asarray(refl).reshape(len(angles), len(wavelengths))


refl = get_tmat()
refl2 = get_liu()
fig, (ax1, ax2) = plt.subplots(2)
#d = np.loadtxt("examples/fabry_filter/fabry.txt")
#ax1.plot(d.T[0], d.T[1]/7000)
for i, a in enumerate(angles):
    ax1.plot(wavelengths, refl[i])
    ax2.plot(wavelengths, refl2[i])
ax1.set_ylim(0,1)
ax1.set_xlim(400,730)
ax2.set_ylim(0,1)
ax2.set_xlim(400,730)
plt.show()

#1.3 * l1 + 2.4 * l2 = 560 / 2 = 280
#1.3 * 
# 130 + 240 = 370 ?= 280
# 520 - 430 = 90nm de cryo 100nm de Zns
# 310-420 = 
