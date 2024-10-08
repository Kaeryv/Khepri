from .crystal import Crystal
from .draw import Drawing
def make_woodpile(rods_w, rods_eps, rods_shift, rods_height, pw, resolution=(256,256)):
    
    pattern = Drawing(resolution, 1)
    pattern.rectangle((0, 0), (1, rods_w), rods_eps)
    
    pattern2 = Drawing(resolution, 1)
    pattern2.rectangle((0,  rods_shift), (1, rods_w), rods_eps)
    pattern2.rectangle((0, -rods_shift), (1, rods_w), rods_eps)

    cl = Crystal(pw)
    cl.add_layer_pixmap("A", pattern.canvas(),    rods_height)
    cl.add_layer_pixmap("B", pattern.canvas().T,  rods_height)
    cl.add_layer_pixmap("C", pattern2.canvas(),   rods_height)
    cl.add_layer_pixmap("D", pattern2.canvas().T, rods_height)

    """
        Define the device and solve.
    """
    device = ['A', 'B', 'C', 'D']
    cl.set_device(device, [False] * len(device))

    return cl
