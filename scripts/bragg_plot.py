import pickle
import numpy as np
points = pickle.load(open("data/bragg_diagram.pkl", "rb"))

import matplotlib.pyplot as plt
fig = plt.figure()
mask = np.logical_and(points[:, 0] > 0, points[:, 0] < 0.01) 
ax = plt.axes(projection='3d', proj_type="ortho")
ax.scatter3D(points[:,0], points[:,1],  points[:,2], 'r.')
ax.view_init(0, 0)
ax.set_xlim(0.0, 0.01)
plt.show()
