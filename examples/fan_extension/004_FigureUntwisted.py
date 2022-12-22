from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-main", action="store_true")
parser.add_argument("-shift", action="store_true")
args = parser.parse_args()

import numpy as np
import matplotlib.pyplot as plt

#T55 = np.load("ds/transmission/untwisted_5x5_001.transmission.npy")
#T33 = np.load("ds/transmission/untwisted_3x3_001.transmission.npy")

#axs[0].plot(freqs, T33, label="3x3")
#axs[0].plot(freqs, T55, label="5x5")
#axs[0].legend()
root = "ds/classic/"

if args.main:
    freqs = np.linspace(0.49, 0.59, 100)
    fig, axs = plt.subplots(5, 1, figsize=(9, 16))
    colors = [ "#a83232", "#a87532", "#a8a432", "#53a832", "#32a875", "#3298a8", "#3244a8"]
    freqs = np.linspace(0.49, 0.59, 300)
    for i in range(7):
        filename = f"{root}/7x7_wi_{i}.npy"
        T = np.load(filename)
        axs[0].plot(freqs, T, color=colors[i])
    axs[0].set_title("7x7 with interface.")
    freqs = np.linspace(0.49, 0.59, 200)

    for i in range(7):
        filename = f"{root}/3x3_{i}.npy"
        T = np.load(filename)
        axs[1].plot(freqs, T, color=colors[i])
    axs[1].set_title("3x3")

    for i in range(7):
        filename = f"{root}/5x5_{i}.npy"
        T = np.load(filename)
        axs[2].plot(freqs, T, color=colors[i])
    axs[2].set_title("5x5")
    freqs = np.linspace(0.49, 0.59, 300)
    for i in range(7):
        filename = f"{root}/7x7_{i}.npy"
        T = np.load(filename)
        axs[3].plot(freqs, T, color=colors[i])
    axs[3].set_title("7x7")
    for i in range(7):
        filename = f"{root}/9x9_{i}.npy"
        T = np.load(filename)
        axs[4].plot(freqs, T, color=colors[i])
    axs[4].set_title("9x9")

    for ax in axs:
        ax.set_xlabel("Frequency ADIM")
        ax.set_ylabel("Transmission")

    plt.tight_layout()
    fig.savefig("Figures/Untwisted.png")

if args.shift:
    freqs = np.linspace(0.49, 0.59, 300)
    T_noshift = np.load(f"{root}/7x7_shifted_00.npy")
    T_shift = np.load(f"{root}/7x7_shifted_01.npy")
    T_noshift_99 = np.load(f"{root}/9x9_shifted_01.npy")
    T_noshift_1313 = np.load(f"{root}/13x13_shifted_01.npy")
    T_noshift_1515 = np.load(f"{root}/21x21_shifted_01.npy")
    fig, ax = plt.subplots()
    ax.plot(freqs, T_noshift)
    ax.plot(freqs, T_shift, label="5x5 noshift")
    ax.plot(freqs, T_noshift_99, label="9x9 noshift")
    ax.plot(freqs, T_noshift_1313, label="13x13 noshift")
    ax.plot(freqs, T_noshift_1515, label="21x21_shifted_01")
    fig.tight_layout()
    fig.legend()
    fig.savefig("Figures/Shifted.png")