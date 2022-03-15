import scipy.linalg as la
import numpy as np
import torch

ng = 30**2
N  = 4 * ng
import time

start =time.time()

a = np.random.rand(N,N)
b = la.expm(a)

stop = time.time()
print(stop - start)

#a = torch.rand(N,N)
#b = torch.matrix_exp(a)