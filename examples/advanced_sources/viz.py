import sys
sys.path.append(".")
from bast.alternative import incident
import numpy as np
pw=(5,5)
P = incident(pw, 1, 1, (0.2,0))
print(P)
P = np.split(P, 2)[0]
import matplotlib.pyplot as plt
F = np.fft.ifft2((P.reshape(pw)))
plt.matshow(np.abs(F))
plt.show()