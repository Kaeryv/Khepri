"""
    A class for designing RCWA's 2D extruded patterns.
    This class can be replaced by you numpy drawing skills.
    Note that:
        arrays are indexed a[x,y].
"""


import numpy as np
from numpy.fft import fft2, fftshift, ifftshift,ifft2
from scipy.interpolate import RegularGridInterpolator
from copy import copy
from collections import namedtuple

Axis = namedtuple('Axis', ['X', 'Y'])(X=[1, 0], Y=[0,1])

def uniform(shape, epsilon=1):
    return np.ones(shape)*epsilon

class Drawing:
    def __init__(self, shape, epsilon_background, lattice=None) -> None:
        self.geometric_description = list()

        if lattice is None:
            self.lattice = lattice = np.array([Axis.X, Axis.Y])
        self.lattice = lattice

        self.background = epsilon_background
        self._canvas = uniform(shape, epsilon=epsilon_background)
        self.xbar = np.linspace(-0.5, 0.5, shape[0])
        self.ybar = np.linspace(-0.5, 0.5, shape[1])

        self.nX, self.nY = np.meshgrid(self.xbar, self.ybar, indexing="ij")
        self.X = lattice[0, 0] * self.nX + lattice[1, 0] * self.nY
        self.Y = lattice[0, 1] * self.nX + lattice[1, 1] * self.nY
        self.x0, self.y0, self.x1, self.y1 = np.min(self.X), np.min(self.Y), np.max(self.X), np.max(self.Y)
        self.x = np.linspace(self.x0, self.x1, shape[0])
        self.y = np.linspace(self.y0, self.y1, shape[1])

    
    def disc(self, xy, radius, epsilon):
        x, y = xy
        self._canvas[np.sqrt((self.X-x)**2+(self.Y-y)**2) < radius] = epsilon
        self.geometric_description.append({"type": "disc", "params": [0.5+xy[0], 0.5+xy[1], radius], "epsilon": epsilon})
    
    def rectangle(self, xy, wh, epsilon):
        x, y = xy
        w, h = wh
        x = x - w / 2
        y = y - h / 2
        xmask = (self.X >= x) & (self.X <= x + w)
        ymask = (self.Y >= y) & (self.Y <= y + h)
        self._canvas[xmask & ymask] = epsilon
        bounds = [0.5 + x, 0.5+y, 0.5+x+w, 0.5+y+h]
        self.geometric_description.append({"type": "rectangle", "params": bounds, "epsilon": epsilon})

    def ellipse(self, xy, ab, r_rad, epsilon):
        x, y = xy
        a, b = ab
        A = (np.cos(r_rad)**2 / a**2) + (np.sin(r_rad)**2 / b**2)
        B = (np.cos(r_rad)**2 / b**2) + (np.sin(r_rad)**2 / a**2)
        C = 2 * np.cos(r_rad) * np.sin(r_rad) * (1/a**2-1/b**2)
        dsqr = (self.X-x)**2 + (self.Y-y)**2
        theta = np.arctan2(self.Y-y, self.X-x)
        irsqr = A*np.cos(theta)**2+B*np.sin(theta)**2+C*np.sin(theta)*np.cos(theta)
        self._canvas[dsqr < 1.0 / irsqr] = epsilon
        self.geometric_description.append({"type": "ellipse", "params": (x,y,a,b,r_rad), "epsilon": epsilon})

    def parallelogram(self, xy, bh, tilt_rad, epsilon):
        '''
        Parallelogram with basis parallel to x axis.
        '''
        x, y = xy
        b, h = bh
        y = y -  h / 2
        tantilt = np.tan(tilt_rad)
        
        for i in range(self._canvas.shape[1]):
            if np.all(self.Y[:, i] <= y):
                continue
            elif np.all(self.Y[:, i] - y >= h):
                break
            else:
                start = x -b /2 + tantilt * self.Y[0, i] #(self.Y[0, i]) / b_over_h + x
                end = start + b
                self._canvas[(self.X[:, i] >start) & (self.X[:, i] <= end), i] = epsilon



    def canvas(self, shape=None, interp_method="linear", return_xy=False):
        if shape is None:
            if return_xy:
                return self._canvas.copy(), self.X, self.Y
            else:
                return self._canvas.copy()
        else:
            XY = np.vstack((self.X.flat,self.Y.flat))
            interp = RegularGridInterpolator((self.x, self.y), self._canvas, method=interp_method)
            xi = np.linspace(self.x0, self.x1, shape[0])
            yi = np.linspace(self.y0, self.y1, shape[1])
            X, Y = np.meshgrid(xi, yi, indexing="ij")
            XY = np.vstack((X.flat, Y.flat)).T
            if return_xy:
                return interp(XY).reshape(shape), X, Y
            else:
                return interp(XY).reshape(shape)

    def islands(self):
        return self.geometric_description

    def plot(self, filename=None, what="dielectric", ax=None, tiling=(1,1), **kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(5,5))

        img, X, Y = self.canvas(**kwargs, return_xy=True)
        if what == "dielectric":
            pass
        elif what == "fourier":
            img = fftshift(fft2(img)) / np.prod(img.shape)
        elif what == "reconstruct":
            fourier = fftshift(fft2(img)) / np.prod(img.shape)
            img = ifft2(ifftshift(fourier)) * np.prod(img.shape)
        
        tx, ty = tiling
        a1, a2 = self.lattice
        for i in np.arange(-(tx//2), (tx//2)+1):
            for j in np.arange(-(ty//2), (ty//2)+1):
                handle = ax.pcolormesh(i * a1[0] + j * a2[0] + X, i * a1[1] + j * a2[1] + Y, img.real, cmap="Blues")
        ax.axis("equal")
        plt.colorbar(handle)

        if filename is not None:
            plt.savefig(filename)

        if filename is None and ax is None:
            plt.show()


        return handle
    
    def from_numpy(self, X):
        '''
            Will convert numpy array geometry to islands description.
            This has limits and only works for 1D array.
        '''
        shape = X.shape
        if len(shape) > 1 and shape[0] > 1 and shape[1] > 1:
            raise NotImplementedError("Currently only supporting 1D gratings.")
        X = X.flatten()
        length = X.shape[0]

        materials = np.unique(X).tolist()
        if len(materials) == 1:
            # If there is only on value it should be the background
            # No need to compute anything
            self.background = materials[0]
            return

        current_material = materials.index(X[0])
        previous_material = copy(current_material)
        background_material = materials.index(self.background)
        current_material_start = 0
        material_intervals = list()
        for i in range(1, length):
            current_material = materials.index(X[i])
            if current_material != previous_material:
                if previous_material != background_material:
                    material_intervals.append((current_material_start, i, materials[previous_material]))
                    current_material_start = copy(i)
                else:
                    current_material_start = copy(i)

                previous_material = copy(current_material)
        
        if current_material != background_material:
            material_intervals.append((current_material_start, length, materials[current_material]))

        material_intervals = np.asarray(material_intervals)
        boundaries = (material_intervals[:, :2] - length / 2) / length
        centers = (boundaries[:,1] + boundaries[:,0]) / 2
        widths  = (boundaries[:,1] - boundaries[:,0])

        for e, c, w in zip(material_intervals[:,-1], centers, widths):
            self.rectangle((c, 0), (w, 1), e)

            
            



        
