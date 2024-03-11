import sys
sys.path.append(".")
import numpy as np
import matplotlib.pyplot as plt
from bast.tmat.lattice import CartesianLattice
from bast.tmat.scattering import scattering_matrix, scattering_interface
from bast.tools import c, incident, compute_fluxes
from bast.alternative import scattering_structured_layer, Lattice, scat_base_transform,redheffer_product,incident as incident2,scattering_reflection,scattering_transmission, scattering_uniform_layer, convolution_matrix
from numpy.lib.scimath import sqrt as csqrt
from tqdm import tqdm
from numpy.linalg import solve, inv


pw = (3,3)
a = 400e-9
lattice = CartesianLattice(pw, (a,0), (0,a))
wl = 300
freq = c / (wl*1e-9)
omega = 2 * np.pi * freq
angle = 0
k0 = 2*np.pi/(wl*1e-9)
kpinc = (0,0)
l = Lattice(pw, a, wl*1e-9, kpinc)
d = 2 * a  

# Define the structure
N = 256
eps = np.ones((N,N))
x = np.linspace(-0.5, 0.5, N, endpoint=True)
y = np.linspace(-0.5, 0.5, N, endpoint=True)
X, Y = np.meshgrid(x, y)
eps[np.sqrt(X**2+Y**2)<= 0.25] = 8.0



S1, ev, WI, VI = scattering_structured_layer(l, eps, d, return_eigenspace=True)
             
#Sr, Wref = scattering_reflection(l.Kx, l.Ky, l.W0, l.V0)
#St, Wtrans = scattering_transmission(l.Kx, l.Ky, l.W0, l.V0)
#
#S = redheffer_product(Sr, S1)
#S = redheffer_product(S, St)

#epsi=1
#kzi = np.conj(csqrt(k0**2*epsi-kpinc[0]**2-kpinc[1]**2))
#esrc = incident2(l.pw, 1, 0, kp=(l.kp[0], l.kp[1], kzi))

#def propagate(e, d, eigenvalues):
#    phase = np.exp(1j * eigenvalues * d)
#    return e * phase

#plt.figure()
#plt.matshow(l.W0.real)
#plt.show()
#@c_trans  = S[1, 0] @ esrc
#eprop_s = propagate(esrc, 0.2*d, k0*np.diag(ev))
#eprop_e = propagate(e_transmitted, d - 0.2*d, k0*np.diag(ev))
#K = np.block([
#    [ l.Ky @ l.Ky ,     -l.Ky @ l.Kx    ],
#    [-l.Kx @ l.Ky,       l.Kx @ l.Kx    ]
#]) # Include epsilon here
#m =  k0**2 * (1-K) @ evec / np.diag(ev)
#FCM = np.block([[m, -m], [evec, evec]])
#A = np.hstack((eprop_s, eprop_e))
S1 = np.asarray(np.bmat([ list(s) for s in list(S1) ]))

svals, svecs = np.linalg.eig(S1)
mask = np.abs(np.abs(svals) -1) < 5e-2
svecs = svecs[:, mask]
svals = svals[mask]
print(np.count_nonzero(mask), "EIG")

index=6
V = svecs[:,index] # Eigenvector i
V2 = svecs[:,index] * svals[index]
print(np.abs(svals[index]))
V = np.split(V, 2)
V2 = np.split(V2, 2)
A = np.hstack((V[1], V2[1]))
W0, V0 = l.W0, l.V0
FCM = np.block([[WI, WI], [-VI, VI]])
FCM0 = np.block([[W0, W0], [-V0, V0]])
fields = inv(FCM) @ FCM0 @ A
exf, eyf, hxf, hyf = np.split(fields, 4)
unit_cells = (2,2)
res = 101
xy = np.linspace(0, a*unit_cells[0], unit_cells[0]*res)
XX, YY = np.meshgrid(xy,xy)
phase = np.exp(1j * (0*XX+0*YY))


def amplitude2field(fourier, shape):
    mid = shape[0] // 2
    i1 = pw[0]//2
    i2 = pw[0]//2+1
    ext_fourier = np.zeros(shape, dtype=np.complex128)
    ext_fourier[mid-i1:mid+i2, mid-i1:mid+i2] = np.fft.fftshift(fourier.reshape(pw))
    return np.fft.ifft2(np.fft.fftshift(ext_fourier))

ey = np.tile(amplitude2field(eyf, (res,res)), unit_cells)#*phase
ex = np.tile(amplitude2field(exf, (res,res)), unit_cells)#*phase
fig, (ax1, ax2) = plt.subplots(1, 2)
im = ax1.matshow(np.real(ex), cmap="RdBu")
ax1.set_title(np.mean(np.abs(np.real(ex))))
#plt.colorbar(im)
im = ax2.matshow(np.real(ey), cmap="RdBu")
ax2.set_title(np.mean(np.abs(np.real(ey))))
#ax1.matshow(np.abs(esrc[9:].reshape(pw)), cmap="RdBu")
#ax2.matshow(np.abs(e_transmitted[:9].reshape(pw)), cmap="RdBu")
#ax1.contour(eps, levels=[2])
#ax2.contour(eps, levels=[2])

plt.savefig("fields.png")


# Transmission
#tx, ty = np.split(Wtrans @ e_transmitted, 2)
#tz = - (l.Kx @ tx + l.Ky @ ty) / np.diag(l.Kz)
#t = np.diag(l.Kz.real/kzi.real*k0.real) * (np.abs(tx)**2+np.abs(ty)**2+np.abs(tz)**2)
#T = np.sum(t)
#
#print(T)
