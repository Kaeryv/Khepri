from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-angle", type=str, default="0:45:45", help="Angles to compute [degrees].")
parser.add_argument("-freq", type=str, default="0.7:0.84:80", help="Reduced frequencies to compute [c/a]")
parser.add_argument("-workdir", type=str, required=True)
parser.add_argument("-a", type=float, default=1e-7)
parser.add_argument("-batch_size", type=int, required=True)
parser.add_argument("-name", type=str, required=True)
args = parser.parse_args()

import numpy as np
import os
from bast.misc import str2linspace_args as linspace_args
from itertools import product
from bast.tools import c

a = args.a
amin, amax, ares = linspace_args(args.angle)
fmin, fmax, fres = linspace_args(args.freq)
angles = np.linspace(amin, amax, ares)
freqs = np.linspace(fmin, fmax, fres) * c / a

inputs = np.asarray(list(product(freqs, angles)))
transmission = np.zeros((inputs.shape[0]))
for i in range(inputs.shape[0]//args.batch_size):
    filename =f"{args.workdir}/T_{i}.npy"
    if os.path.isfile(filename):
        t =np.load(filename)
        transmission[i*args.batch_size:(i+1)*args.batch_size] = t
    else:
        transmission[i*args.batch_size:(i+1)*args.batch_size] = 0.0
np.save(f"ds/transmission/{args.name}.transmission.npy", transmission)
