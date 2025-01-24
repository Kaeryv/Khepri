from cmath import pi

c = 299792458.0
mu0 = 4 * pi * 1e-7
mu0c = mu0 * c
twopi = 2 * pi
h = 6.6261e-34
ev = 1.602e-19

def fn2ev(f, pitch):
    return h * f * c / (pitch * 1e-6) / ev
def ev2fn(fev, pitch):
    return fev / h / c * (pitch * 1e-6) * ev