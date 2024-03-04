'''
Visualisation de la combinaison des matrices S.
'''
import sys
sys.path.append(".")
from bast.tools import joint_subspace, as_strided, block2dense
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    os.makedirs("Figures/", exist_ok=True)
    N = 3
    S = np.zeros((2, 2, 2*N, 2*N), dtype=np.complex128)
    S4 = S.reshape(2,2,2,N, 2, N)
    S4 = np.swapaxes(S4, 3, 4).reshape(4,4, N, N)
    ds = S.strides[-1]
    # Each polarization block gets its color
    for i in range(4):
        for j in range(4):
            c = i * 4 + j
            S4[i, j] = 1j*(1+c)
    S4[0, 0] = 1j*np.random.permutation(np.arange(N**2)).reshape((N,N))
    S4 = S4.reshape(2,2,2,2, N, N)
    S4 = np.swapaxes(S4, 3, 4).reshape(2,2,2*N,2*N)
    S2 = joint_subspace([S4]*N, kind=1)
    fig, (ax1, ax2) = plt.subplots(1, 2,  figsize=(10,6))
    print(S.shape)
    ax1.matshow(block2dense(S4.imag), cmap="tab20", vmin=0, interpolation=None, vmax=16)
    ax2.matshow(block2dense(S2.imag), cmap="tab20", vmin=0, vmax=16)
    for ax, size in zip((ax1, ax2), [4*N, 4*N**2]):
        ax.set_xticks(np.arange(size)-0.5, map(str, range(size)))
        ax.set_yticks(np.arange(size)-0.5, map(str, range(size)))
        ax.grid(color='k', linewidth=1)
        ax.set_xticklabels([])
    plt.savefig("Figures/JointSubspaces.png")
