from bast.scattering import scattering_matrix
from bast.lattice import CartesianLattice
from bast.fourier import transform
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == "__main__":
    os.makedirs("Figures/", exist_ok=True)
    pw = (7, 7)
    a=1e-7
    l = CartesianLattice(pw, (a, 0), (0, a), 1.0, 1.0)
    bf = transform("disc", [0.5*a, 0.5*a, 0.25*a], l.Gx, l.Gy, l.area)

    struct = np.fft.ifft2(bf)
    plt.matshow(np.abs(struct))
    plt.savefig("Figures/Structure.png")