from argparse import ArgumentParser

from bast.tmat.scattering import scattering_matrix
parser = ArgumentParser()
parser.add_argument("-input", type=str, required=True)
parser.add_argument("-range", type=int, nargs="+", required=True)
parser.add_argument("-logname", type=str, required=True)
args = parser.parse_args()

import logging
logging.basicConfig(filename=f"logs/log_{args.logname}.{args.input}.txt", level=logging.INFO)
logging.info(f"Starting computation of interlayers from {args.range[0]} to {args.range[1]}")

import os
import json


if os.path.isfile(args.input):
    with open(args.input, "r") as f:
        config = json.load(f)["args"]
def str2val(string):
    s = string.split(":")
    min = float(s[0])
    max = float(s[1])
    res = int(s[2])
    return min, max, res

import numpy as np

from bast.tmat.lattice import CartesianLattice
from bast.tools import incident, c

from itertools import product
from tqdm import tqdm

amin, amax, ares = str2val(config["angle"])
fmin, fmax, fres = str2val(config["freq"])
M, N = ares, fres
pw = config["pw"]
a = 1e-7
angles = np.linspace(amin, amax, M)
freqs = np.linspace(fmin, fmax, N) * c / a

def get_l(angle, pw):
    l1 = CartesianLattice(pw, (a, 0), (0., a), 1.0, 1.0)
    l2 = CartesianLattice(pw, (a, 0), (0., a), 1.0, 1.0)
    l2.rotate(angle)
    l = l1 + l2
    return l


print(f"Starting computations for from {args.range[0]} {args.range[1]}")
inputs = np.asarray(list(product(freqs, angles)))
for i in range(*args.range):
    logging.info(f"@ iteration {i}")
    freq, angle = inputs[i, :]
    wl = c / freq
    l = get_l(angle, pw=pw)
    mask = l.restrict_radius_mask(2e9)
    S_interlayer, _ = scattering_matrix(l.gx.shape, l, "uniform", [], 1.0, 1.0, wl, kp=(0,0),
        depth=0.3*a, mask=mask)
    np.savez_compressed(f"ds/interlayers/interlayer_7x7/Ss_{i:05d}", Si=S_interlayer, a=a)
