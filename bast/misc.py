def str2linspace_args(string):
    s = string.split(":")
    mn = float(s[0])
    mx = float(s[1])
    ct = int(s[2])
    return mn, mx, ct
