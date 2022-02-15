from .tools import compute_eta, compute_kplanar, reciproc, gvectors
from .tools import compute_mu, unitcellarea, compute_kz
from .matrices import matrix_u, matrix_v

"""
Lattice class

    This class is used to store the lattice parameters and 
    the reciprocal lattice vectors.

    The information about the incidence and emergence media is also stored
    The class provides methods to compute the polarization basis transformation
"""
class CartesianLattice:
    def __init__(self, pw, a1, a2, eps_incid=1.0, eps_emerg=1.0) -> None:
        b1, b2 = reciproc(a1, a2)
        self.n = tuple(p // 2 for p in pw)
        self.q = tuple(4 * i for i in self.n)
        self.Gx, self.Gy = gvectors(b1, b2, self.q)
        self.area = unitcellarea(a1, a2)
        self.eps_incid = eps_incid
        self.eps_emerg = eps_emerg
        self.eta_ = None

        self.kzi_ = dict()

    @property
    def gx(self):
        """
        The x-component of the reciprocal lattice vectors
        """
        return self.Gx[self.n[0]:3*self.n[0], self.n[1]:3*self.n[1]]
    
    @property
    def gy(self):
        """
        The y-component of the reciprocal lattice vectors
        """
        return self.Gy[self.q[0]//4:3*self.q[0]//4,self.q[1]//4:3*self.q[1]//4]

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
        self.kzi_[wavelength] = compute_kz(self.gx, self.gy, self.eps_incid, wavelength, kp)
        return self.kzi_[wavelength]

    def kze(self, wavelength, kp):
        """
        The emergent wavevector
        """
        return compute_kz(self.gx, self.gy, self.eps_emerg, wavelength, kp)

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
            self.eps_incid, wavelength)
    
    def Vi(self, wavelength, kp):
        """
        The polarization basis transformation matrix
        """
        return matrix_v(
            self.eta(wavelength, kp), 
            self.mugi(wavelength, kp),
            self.eps_incid)

    def Ve(self, wavelength, kp):
        """
        The polarization basis transformation matrix
        """
        return matrix_v(
            self.eta(wavelength, kp), 
            self.muge(wavelength, kp),
            self.eps_emerg)

