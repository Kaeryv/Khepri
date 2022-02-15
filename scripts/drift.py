import sys
sys.path.append('../hybris/')
sys.path.append('.')

from hybris.optimizer import Optimizer
from hybris.ext_problem import Problem
from hybris.backend import CTestFunction
import numpy as np
import ctypes
from ctypes import c_double
from ctypes import CFUNCTYPE, c_int, c_void_p
import numpy as np
from bast.eigentricks import scattering_splitlr as split

for i in range(100):
    S = np.load(f"data/S_drift_{i}.npy")
    Sl, Sr = split(S)
    best =np.inf
    angle = 0.0
    #p = Problem()
    def f(pos, n, d, fit, state):
        global best
        global angle
        x = np.ctypeslib.as_array(pos, (n, d))
        y = np.ctypeslib.as_array(fit, (n,))
        for i in range(n):
            k_angle = x[i, 0]
            P = Sl - np.exp(1j * k_angle) * Sr
    
            sol = np.log1p(np.abs(np.linalg.det(P) ))
            y[i] = sol
            if sol < best:
                best = sol
                angle = k_angle
    
    
    fc = CTestFunction(f)
    
    p = Problem(c_double(0.0), c_double(3.1415), 0, fc, "test")
    
    for j in range(5):
        best =np.inf
        opt = Optimizer(10, 1, 100)
        opt.init(p)
        opt.minimize(p)
        if opt.profile[-1] < 1e-3:
            print(angle)
        else:
            print("NaN")
    #import matplotlib.pyplot as plt
    #plt.plot(opt.profile)
    #plt.show()
    #print(opt.profile[-1])
    #x = np.linspace(0, 1, 20)
    #X, Y = np.meshgrid(x, x)
    #Z = f(X, Y)
    
    #import matplotlib.pyplot as plt
    #
    #plt.matshow(Z, extent=[0, 1, 0, 1])
    #plt.show()
    
