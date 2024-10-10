from PIL import Image
import numpy as np
raw = Image.open("fan21.png")
raw = np.asarray(raw).astype(np.float64)
import matplotlib
import matplotlib.pyplot as plt

cmap = matplotlib.cm.get_cmap('viridis')

def find_value_from_color(rgb_val):
    tol = 8e-3
    for val in np.linspace(0.0, 1.0, 255):
        rgba = cmap(val)
        distance = np.mean(np.power(rgba-rgb_val,2))
        if distance < tol:
            return 0.1+val *.75

print(raw.shape)
map = np.empty((raw.shape[0]))
from tqdm import trange
#for i in trange(raw.shape[0]):
#    for j in range(raw.shape[1]):
#        map[i,j] = find_value_from_color(raw[i,j])
#plt.matshow(map)
plt.show()
plt.figure()
freq = np.flip(np.linspace(0.7, 0.84, map.shape[0]))
data = np.loadtxt("rta.txt")

for j in range(raw.shape[0]):
    map[j] = find_value_from_color(raw[j,40]/255.)
plt.plot(freq, map)
plt.plot(1e-6 / data[:,0], data[:, 2])

plt.show()