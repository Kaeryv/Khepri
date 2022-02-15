import numpy as np
Sli  = np.load("./data/600nm/Sli.npy")[::4,::4]
print(Sli.shape)
Slr  = np.load("./data/600nm/Slr.npy")[::4,::4]
C = Sli + Slr

