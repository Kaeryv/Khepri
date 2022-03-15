from .tools import compute_eta, compute_kplanar, grid_size, reciproc, gvectors
from .tools import compute_mu, unitcellarea, compute_kz
from .matrices import matrix_u, matrix_v
import numpy as np

complex_dtype = { np.float32: np.complex64, np.float64: np.complex128 }

"""
Lattice class

    This class is used to store the lattice parameters and 
    the reciprocal lattice vectors.

    The information about the incidence and emergence media is also stored
    The class provides methods to compute the polarization basis transformation
"""
class CartesianLattice:
    def __init__(self, pw, a1, a2, eps_incid=1.0, eps_emerg=1.0, dtype=np.float32):
        b1, b2 = reciproc(a1, a2)
        self.b1 = b1
        self.b2 = b2
        self.n, self.q = grid_size(pw)
        self.Gx, self.Gy = gvectors(b1, b2, self.q, dtype=dtype)
        self.area = unitcellarea(a1, a2)
        self.eps_incid = eps_incid
        self.eps_emerg = eps_emerg
        self.eta_ = None

        self.kzi_ = dict()

        self.dtype = dtype

    @property
    def gx(self):
        """
        The x-component of the reciprocal lattice vectors
        """
        nx = (self.q[0] // 4, (self.q[0] - 1) // 4)[self.q[0] % 2]
        ny = (self.q[1] // 4, (self.q[1] - 1) // 4)[self.q[1] % 2]
        return self.Gx[nx:self.q[0]-nx, ny:self.q[0]-ny]
    
    @property
    def gy(self):
        """
        The y-component of the reciprocal lattice vectors
        """
        nx = (self.q[0] // 4, (self.q[0] - 1) // 4)[self.q[0] % 2]
        ny = (self.q[1] // 4, (self.q[1] - 1) // 4)[self.q[1] % 2]
        return self.Gy[nx:self.q[0]-nx, ny:self.q[0]-ny]

    @property
    def eta_normal(self):
        """
        Eta for normal incidence
        """
        if self.eta_ is None:
            self.eta_ = compute_eta(self.gx, self.gy)
        return self.eta_
     
    def eta(self, wavelength, kp):
        """
        Eta for a given angle and wavelength
        """
        return compute_eta(self.gx, self.gy, kp=kp)

    
    def kp_angle(self, wavelength, theta_deg, phi_deg):
        """
        The planar wavevector
        """
        kp = compute_kplanar(self.eps_incid, wavelength, theta_deg, phi_deg)
        return kp


    def kzi(self, wavelength, kp):
        """
        The incident wavevector
        """
        #if wavelength in self.kzi_.keys():
        #    return self.kzi_[wavelength]
        #else:
        return compute_kz(self.gx, self.gy, self.eps_incid, wavelength, kp, complex_dtype[self.dtype])

    def kze(self, wavelength, kp):
        """
        The emergent wavevector
        """
        return compute_kz(self.gx, self.gy, self.eps_emerg, wavelength, kp, complex_dtype[self.dtype])

    def mugi(self, wavelength, kp):
        return compute_mu(self.gx, self.gy, self.kzi(wavelength, kp), self.eps_incid, wavelength, kp)

    def muge(self, wavelength, kp):
        return compute_mu(self.gx, self.gy, self.kze(wavelength, kp), self.eps_emerg, wavelength, kp)

    def U(self, wavelength, kp):
        """
        The polarization basis transformation matrix
        """
        return matrix_u(self.kzi(wavelength, kp), 
            self.eta(wavelength, kp), 
            self.mugi(wavelength, kp), 
            self.eps_incid, wavelength,
            complex_dtype[self.dtype])
    
    def Vi(self, wavelength, kp):
        """
        The polarization basis transformation matrix
        """
        return matrix_v(
            self.eta(wavelength, kp), 
            self.mugi(wavelength, kp),
            self.eps_incid, 
            complex_dtype[self.dtype])

    def Ve(self, wavelength, kp):
        """
        The polarization basis transformation matrix
        """
        return matrix_v(
            self.eta(wavelength, kp), 
            self.muge(wavelength, kp),
            self.eps_emerg, 
            complex_dtype[self.dtype])

    def bz_path(self, k_points, resolutions=[], a=1.0):
        main_k_points = { 'G': (0, 0), 'X': (0, 0.99*np.pi /a ), 'M': (0.99*np.pi/a, 0.99*np.pi/a) }
        # Return a path of Bz vectors
        assert len(k_points) > 1, "Give at least two k-points"
        if len(resolutions) == 0:
            resolutions = [10]*(len(k_points)-1)
        elif len(resolutions) == 1:
            resolutions = [resolutions[0]]*(len(k_points)-1)
        assert len(resolutions) == len(k_points)-1, "Give resolution for each interval between k-points"

        traj = []
        for i in range(len(resolutions)):
            start = main_k_points[k_points[i]]
            stop = main_k_points[k_points[i+1]]
            for j in range(resolutions[i]):
                traj.append(np.array([start[0] + (stop[0] - start[0])*j/resolutions[i], start[1] + (stop[1] - start[1])*j/resolutions[i]]))

        return traj