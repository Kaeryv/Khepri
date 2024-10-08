from typing import List, Tuple
from scipy.linalg import eig
import numpy as np

def scattering_splitlr(S) -> Tuple[np.array, np.array]:
    # For generalized eigenvalue problem
    ng = S.shape[0] // 4

    p = np.s_[0:2*ng]
    m = np.s_[2*ng:4*ng]

    Sl = np.zeros_like(S)
    Sl[p, p] = + S[p, p]
    Sl[m, p] =   S[m, p]
    Sl[m, m] = - np.eye(2*ng, 2*ng, dtype=Sl.dtype)

    Sr = np.zeros_like(S)
    Sr[p,p] = + np.eye(2*ng, 2*ng, dtype=Sr.dtype)
    Sr[p,m] = - S[p,m]
    Sr[m,m] = - S[m,m]

    return Sl, Sr

def scattering_det(S):
    ng = S.shape[0] // 4
    p = np.s_[0:2*ng]
    m = np.s_[2*ng:4*ng]
    return np.linalg.det(S[p,p] - S[p,m] @ np.linalg.inv(S[m,m]) @ S[m,p]) * np.linalg.det(S[m,m])
def scattering_eigenvalues(S, dos=False):
    # Scattering to generalized eigenvalue problem
    Sl, Sr = scattering_splitlr(S)
    #iSr = np.linalg.inv(Sr)
    if np.any(np.isnan(Sr)) or np.any(np.isnan(Sl)):
        return None
    #w, v = eig(iSr @ Sl)
    w, v = eig(Sl, Sr)
    if dos:
        return w, v, scattering_det(S)
    else:
        return w, v

def on_shell(eigenvalues, tol=1e-10):
    return np.isclose(np.abs(eigenvalues), 1.0, rtol=0.0, atol=tol)

from itertools import chain

def band_structure(S: List[np.array]):
    w, v = scattering_eigenvalues(S)
    if w is None:
        return None
    w = w.real
    w = w[w > 0]
    w = w[w < 1]
    w = w[on_shell(w)]
    return w
