from argparse import ArgumentParser
import logging
logging.basicConfig(filename='example.log', level=logging.INFO) #encoding='utf-8', 
logging.getLogger().addHandler(logging.StreamHandler())
parser = ArgumentParser()
parser.add_argument("-n", "--magic_angle", required=True, type=int)
parser.add_argument("-d", "--dry_run", action='store_true')
parser.add_argument("-g", "--graphics", action='store_true')
parser.add_argument("-m", "--multiplier", default=3.0, type=float)
parser.add_argument("-p", "--pad", default=40.0)
parser.add_argument("-name", required=True)
args = parser.parse_args()

from math import prod, floor, ceil, asin, sqrt, pi, tan, sin, radians, degrees, cos
i = args.magic_angle

def magic_angle_square(p, q=1):
    return asin(p*q/(p**2+q**2))
#theta = 2 * asin(asin(1./(2*sqrt(3*i*i+3*i+1))));
theta = magic_angle_square(i)
logging.info(f"Creating a structure for angle theta_{args.magic_angle} = {round(degrees(theta), 4)} degrees.")

def supercell_multiplicity(theta, gamma=3, delta=1):
    return sqrt(2 / gamma**2 / delta / sin((theta/2))**2)

freq = round(supercell_multiplicity(theta))
    
mul = args.multiplier
a = 6 * mul
pad = a * float(args.pad)
n = round(freq * a + pad)
shape = (n, n)

logging.info(f"The fourier space shape required is {(shape[0]-pad, shape[1]-pad)} with {freq} multiplicity")

if args.dry_run:
    exit()

import numpy as np
from PIL import Image, ImageDraw

im = Image.fromarray(np.zeros(shape))
draw = ImageDraw.Draw(im)
for i in np.arange(a/2, shape[0], a):
    for j in np.arange(a/2, shape[1], a):
        r = 2.0 * mul
        draw.ellipse((i-r+0.5, j-r+0.5, i+r-0.5, j+r-0.5), fill = 'blue', outline ='blue')
from numpy.matlib import repmat

def normalize(img):
    img -= img.min()
    img /= img.max()
    return img
im_not_twisted = normalize(np.array(im).copy().astype(np.float64))
im = im.rotate(degrees(theta), resample=Image.BICUBIC, expand=False)
im_twisted = normalize(np.array(im).copy().astype(np.float64))
pad2 = round(pad / 2)
im_twisted = im_twisted[pad2:-pad2, pad2:-pad2]
im_not_twisted = im_not_twisted[pad2:-pad2, pad2:-pad2]

print(im_not_twisted.max(), im_not_twisted.min())
fft_twisted = np.fft.fftshift(np.fft.fft2(im_twisted))
fft_not_twisted = np.fft.fftshift(np.fft.fft2(im_not_twisted))
fft_twisted /= prod(fft_twisted.shape)
fft_not_twisted /= prod(fft_not_twisted.shape)

if args.graphics:
    import matplotlib.pyplot as plt
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.matshow(repmat(im_not_twisted, 2, 2), extent=[0, 2, 0, 2], alpha=0.7)
    ax1.matshow(repmat(im_twisted, 2, 2), extent=[0, 2, 0, 2], alpha=0.7)
    rectangle = plt.Rectangle((0,0), 1, 1, fc=(0,0,0,0.7),ec="black")
    ax1.text(0.1, 0.1, "Unit Cell", size=24,color="w")
    ax1.add_patch(rectangle)
    ax1.set_title(f"2x2 Bilayer twisted unit cells\n by {round(degrees(theta),2)} degrees.")
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    h = ax2.matshow(np.log1p(np.abs(fft_twisted)), extent=[0, 1, 0, 1])
    ax2.set_title("1x1 Unit cell Fourier transform")
    fig.colorbar(h, cax=cax, orientation='vertical')
    cax.set_ylabel("Log Magnitude of FFT")
    plt.tight_layout()
    plt.savefig("figures/structure.png")
    plt.show()

# Save the stuff
import h5py as h5
hf = h5.File(f"./data/{args.name}.hdf5", "w")
hf['struct/real/twisted'] = im_twisted
hf['struct/real/not_twisted'] = im_not_twisted
hf['struct/fft/not_twisted'] = fft_not_twisted
hf['struct/fft/twisted'] = fft_twisted
hf['struct'].attrs["multiplicity"] = int(freq)
hf['struct'].attrs["pw"] = (int((shape[0]-pad)//2), int((shape[0]-pad)//2))
hf['struct'].attrs["angle_radians"] = theta
hf.close()

'''
im = im.resize((round(im.size[0]//ds)-4, round(im.size[1]//ds)-4), resample=Image.ANTIALIAS)
'''
#im = Image.fromarray(np.zeros(shape))
#draw = ImageDraw.Draw(im)
#for i in np.arange(a/2, shape[0], a):
#    for j in np.arange(a/2, shape[1], a):
#        x = i* cos(theta) - j * sin(theta)
#        y = i * sin(theta) + j* cos(theta)
#        r = 2.0 * mul
#        draw.ellipse((x-r+0.5, y-r+0.5, x+r-0.5, y+r-0.5), fill = 'blue', outline ='blue')

