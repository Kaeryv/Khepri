from argparse import ArgumentParser

from bast.scattering import scattering_matrix
from PIL import Image
parser = ArgumentParser()
parser.add_argument("-figname", type=str, required=True)
args = parser.parse_args()

import matplotlib.pyplot as plt
#plt.rcParams["font.family"] = "DejaVu"
plt.rcParams["font.size"] = 14

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
os.makedirs("Figures/", exist_ok=True)

T = np.load("ds/transmission/kpoints_7x7_001.transmission.npy")
T2 = np.load("ds/transmission/twist_5x5_001.transmission.npy")
T3 = np.load("ds/transmission/highres_5x5_003.transmission.npy")
#T2 = np.load("ds/transmission/fan_3x3_001.transmission.npy")
#T2 = np.load("ds/transmission/5x5_003.transmission.npy")
#T3 = np.load("ds/transmission/fan_3x3_debug.transmission.npy")
#T4 = np.load("ds/transmission/7x7_001.transmission.npy")
M=45
N=100
def matshow_T(ax, T, M=M, N=N):
    T = np.asarray(T).reshape(N,M)
    T[np.isnan(T)]=0.0
    im= ax.matshow(T, origin="lower", vmin=0.0, vmax=1.0,extent=[0, 45, 0.7, 0.84])
    ax.axis("auto")
    ax.xaxis.tick_bottom()
    ax.set_xlabel("Angle [degrees]")
    ax.set_ylabel("Frequency [c/a]")

    return im


#corr = Image.open("Figures/fig2_corr.png")
fig, ((ax,ax2),(ax3,ax4)) = plt.subplots(2, 2,figsize=(12,8))
print(np.max(T))
ax.set_title("Fan")
ax2.set_title("3x3")
ax3.set_title("5x5")
ax4.set_title("7x7")
im  = matshow_T(ax2,T)
im2 = matshow_T(ax3,T2)
im3 = matshow_T(ax4,T3, M=180, N=300)
#ax.matshow(corr, extent=[0, 45, 0.7, 0.84])
ax.axis("auto")
ax.xaxis.tick_bottom()
ax.set_xlabel("Angle [degrees]")
ax.set_ylabel("Frequency [c/a]")

for a in (ax, ax2, ax3, ax4):
  divider = make_axes_locatable(a)
  cax = divider.append_axes('right', size='5%', pad=0.05)
  fig.colorbar(im, cax=cax, orientation='vertical')
plt.title("Transmission")
plt.tight_layout()
plt.savefig(f"Figures/TransmissionFig2_{args.figname}.png")

from mpl_toolkits.axes_grid1 import make_axes_locatable

fig2, ax2 = plt.subplots()
im1 = ax2.matshow(T3.reshape(300, 180), origin="lower", vmin=0.0, vmax=1.0,extent=[0, 45, 0.7, 0.84])
ax2.axis("auto")
ax2.xaxis.tick_bottom()
ax2.set_xlabel("Angle [degrees]")
ax2.set_ylabel("Frequency [c/a]")
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig2.colorbar(im1, cax=cax, orientation='vertical')
fig2.savefig("Figures/FanHighres.png")