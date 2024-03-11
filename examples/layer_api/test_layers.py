import sys
sys.path.append(".")
from bast.expansion import Expansion
from bast.layer import Layer
from bast.crystal import Crystal
import numpy as np
pw = (5,5)
e = Expansion(pw)
#l = Layer.uniform(e, 7.0, 0.3)
#l.solve((0,0), 1/0.7)
l = Layer.pixmap(e, )
c = Crystal(pw, expansion=e)
c.add_layer("S1", l)
c.set_stacking(["S1"])

spectrum = list()
for f in np.linspace(0.0, 1.0):
    c.set_source(1/f, 1, 1, 0,0)
    c.solve()
    R, T=  c.poynting_flux_end()
    spectrum.append((R,T))

import matplotlib.pyplot as plt
plt.plot(spectrum)
plt.show()

