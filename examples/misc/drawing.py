import sys
sys.path.append(".")
from bast.draw import Drawing
import matplotlib.pyplot as plt

canvas_size = 64, 64
d = Drawing(canvas_size, 2)
d.disc((0,0), 0.4, 3)
d.ellipse((0,0), (0.2,0.4), 0, 4)
d.ellipse((0,0), (0.2,0.4), 3.1415/2, 4.5)
d.ellipse((0,0), (0.2,0.4), 3.1415/4, 4.7)
d.rectangle((0.2,0), (0.4,0.2), 5)

d.plot()
# fig, ax = plt.subplots(figsize=(5,5))
# ax.matshow(d.canvas(), cmap="Blues", extent=[0,1,0,1])
# ax.axis('equal')
# plt.show()


