import numpy as np

def str2linspace_args(string):
    s = string.split(":")
    assert len(s) == 3
    mn = float(s[0])
    mx = float(s[1])
    ct = int(s[2])
    return mn, mx, ct

def block_split(S):
    shape = S.shape
    m = shape[0] // 2
    one = slice(0,m)
    two = slice(m,2*m)
    # B = [np.hsplit(half, 2) for half in np.vsplit(S, 2)]
    return np.asarray([[S[one, one], S[one, two]],[S[two, one], S[two, two]]])
