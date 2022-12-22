'''
Visualisation de la combinaison des matrices S.
'''

from bast.tools import joint_subspace, as_strided
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    os.makedirs("Figures/", exist_ok=True)
    N = 4
    S = np.zeros((4*N, 4*N), dtype=np.complex128)
    ds = S.strides[-1]
    viewS = as_strided(S, (4, 4, N, N), (ds*e for e in (N*4*N,N, 4*N, 1)))

    # Each polarization block gets its color
    for i in range(4):
        for j in range(4):
            c = i * 4 + j
            viewS[i, j] = 1j*(1+c)
    viewS[0, 0] = 1j*np.random.randint(0, 16, (4,4))
    S2 = joint_subspace([ S*0+np.nan,  S*0+np.nan,S,S], kind=1)
    fig, (ax1, ax2) = plt.subplots(1, 2,  figsize=(10,6))
    ax1.matshow(S.imag, cmap="tab20", vmin=0, interpolation=None)
    ax2.matshow(S2.imag, cmap="tab20", vmin=0)
    for ax, size in zip((ax1, ax2), [4*N, 4*N**2]):
        ax.set_xticks(np.arange(size)-0.5, map(str, range(size)))
        ax.set_yticks(np.arange(size)-0.5, map(str, range(size)))
        ax.grid(color='k', linewidth=1)
        ax.set_xticklabels([])
    plt.savefig("Figures/JointSubspaces.png")
