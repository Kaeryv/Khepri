import numpy as np
from scipy.interpolate import RegularGridInterpolator
def uniform(shape, epsilon=1):
    return np.ones(shape)*epsilon



class Drawing:
    def __init__(self, shape, epsilon, lattice) -> None:
        self._canvas = uniform(shape, epsilon=epsilon)
        self.x0, self.y0, self.x1, self.y1 = -0.5, -0.5, 0.5, 0.5
        self.xbar = np.linspace(-0.5, 0.5, shape[0])
        self.ybar = np.linspace(-0.5, 0.5, shape[1])
        #self.xaxis = np.linspace(self.x0, self.x1, shape[0], endpoint=True)
        #self.yaxis = np.linspace(self.y0, self.y1, shape[1], endpoint=True)
        self.nX, self.nY = np.meshgrid(self.xbar, self.ybar, indexing="ij")
        self.X = lattice[0, 0] * self.nX + lattice[1, 0] * self.nY
        self.Y = lattice[0, 1] * self.nX + lattice[1, 1] * self.nY
    

    def circle(self, xy, radius, epsilon):
        x, y = xy
        self._canvas[np.sqrt((self.X-x)**2+(self.Y-y)**2) <= radius] = epsilon
    
    def rectangle(self, xy, wh, epsilon):
        x, y = xy
        w, h = wh
        x = x - w / 2
        y = y - h / 2
        xmask = (self.X >= x) & (self.X <= x + w)
        ymask = (self.Y >= y) & (self.Y <= y + h)
        self._canvas[xmask & ymask] = epsilon


    def canvas(self, shape=None, interp_method="linear"):
        if shape is None:
            return self._canvas.copy()
        else:
            XY = np.vstack((self.X.flat,self.Y.flat))
            print(XY.shape)
            interp = RegularGridInterpolator((self.xbar, self.ybar), self._canvas, method=interp_method)
            xi = np.linspace(self.x0, self.x1, shape[0], endpoint=True)
            yi = np.linspace(self.y0, self.y1, shape[1], endpoint=True)
            X, Y = np.meshgrid(xi, yi)
            XY = np.vstack((X.flat, Y.flat)).T
            print(XY.shape)
            return interp(XY).reshape(shape)

    def plot(self, filename=None, what="dielectric", ax=None, **kwargs):
        import matplotlib.pyplot as plt
        img = self.canvas(**kwargs)
        if ax is None:
            fig, ax = plt.subplots()
        ax.matshow(img, origin="lower")
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)



        
