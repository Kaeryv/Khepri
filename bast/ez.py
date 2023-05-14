if __name__ != "__main__":
    print("This module is an executable one, launch using 'python -m bast.ez'")
    exit()

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-s", "--study", required=True)
parser.add_argument("-p", "--project", required=True)
args = parser.parse_args()

import json
from copy import deepcopy as copy
from itertools import product
from types import SimpleNamespace

from tqdm import tqdm
import numpy as np

from .tools import incident, compute_fluxes
from .tools import c
from .scattering import scattering_matrix_islands
from .lattice import CartesianLattice

with open(args.project, "r") as f:
    config = SimpleNamespace(**json.load(f))

materials = config.materials

class Material:
    def __init__(self):
        pass

    def load_state_dict(self, state_dict):
        self._epsilon = state_dict["epsilon"]
        self._color = state_dict["color"]

    @property
    def epsilon(self):
        return self._epsilon 

def json_linspace(json):
    return np.linspace(json["start"], json["end"], json["steps"])

def load_materials(state_dict):
    materials = dict()
    for mat in state_dict:
        materials[mat] = Material()
        materials[mat].load_state_dict(config.materials[mat])
    
    return materials

def load_layers(state_dict):
    '''
    Load the layers from the dictionary.
    Removes some user-friendly notations/segregations
    '''
    layers = dict()

    for layer in config.layers:
        layers[layer] = copy(config.layers[layer])
        this_layer = layers[layer]
        if not "geometry" in this_layer:
            this_layer["geometry"] = list()
        this_layer["raw_geometry"] = list()
        
        for island in this_layer["geometry"]:
            shape = island["shape"]
            # Gather params in raw list
            island_params = copy(island["center"])
            if "radius" in island:
                island_params.append(island["radius"])
            elif "size" in island:
                island_params.extend(island["size"])
            island_mat = island["material"]
            this_layer["raw_geometry"].append({"shape": shape, "adim_params": island_params, "material": island_mat})
    return layers

def compute_spectrum(layers, materials, params, device):
    a = config.lattice["lattice_size"]
    a1 = [x*a for x in config.lattice["a1"]]
    a2 = [x*a for x in config.lattice["a2"]]
    pw = config.lattice["pw"]
    s_pol = params["E0"]["s_real"] + params["E0"]["s_imag"] * 1j
    p_pol = params["E0"]["p_real"] + params["E0"]["p_imag"] * 1j
    
    # Resolve materials
    lattice = CartesianLattice(pw,
        a1=a1, a2=a2, 
        eps_incid=materials[config.lattice["material_incident"]].epsilon,
        eps_emerg=materials[config.lattice["material_substrate"]].epsilon)

    pin = incident(pw, p_pol=p_pol, s_pol=s_pol)

    # Helper function to run trough S-matrices dependencies and locate required computations
    def gather_s(name, required_layers=dict()):
        if name in config.stacking:
            for l in config.stacking[name]["layers"]:
                gather_s(l, required_layers)
        elif name in config.layers:
            required_layers[name] = layers[name]
        return required_layers

    required_layers = gather_s(device)

    assert len(required_layers) > 0, "No computations required."


    rta = list()
    configs = list(product(
        json_linspace(params["phi"]), 
        json_linspace(params["theta"]),
        json_linspace(params["frequency"])))
    for i, (phi, theta, freq) in enumerate(tqdm(configs)):
        wl =  a / freq
        for sconf in required_layers:
            cur_layer = required_layers[sconf]

            # Manage potential dispersive materials and scale dimensions
            host_eps = materials[cur_layer["material"]].epsilon

            for island in cur_layer["raw_geometry"]:
                island["epsilon"] = materials[island["material"]].epsilon
                island["params"] = [ p * a for p in island["adim_params"]]
            kp = lattice.kp_angle(wl, theta, phi)
            cur_layer["S"] = scattering_matrix_islands(pw,lattice, cur_layer["raw_geometry"], 
                host_eps, wavelength=wl, kp=kp,
                depth=cur_layer["depth"] * a, slicing_pow=3)
        
        # Stack the S-matrices
        def stack_s(name):
            if name in required_layers:
                return required_layers[name]["S"]
            elif name in config.stacking:
                if "repeat" in config.stacking[name]:
                    repeat = config.stacking[name]["repeat"]
                else:
                    repeat = 1
                
                S0 = stack_s(config.stacking[name]["layers"][0])
                for ss in config.stacking[name]["layers"][1:]:
                    S0 = S0 @ stack_s(ss)

                if repeat > 1:
                    Srepeat = S0.copy()
                    for i in range(repeat-1):
                        Srepeat = Srepeat @ S0
                    return Srepeat
                else:
                    return S0
    
        Stot = stack_s(device)
        S_interface = scattering_interface(lattice, wl, kp=kp)
        Stot = Stot @ S_interface
        pout = Stot @ pin
        R, T, A = compute_fluxes(lattice, wl, pin, pout, kp=kp)

        rta.append((phi, theta, freq, wl, R, T, A))

    if "output_figure" in params:
        fig, ax = plt.subplots()
        ax.plot(rta)
        fig.savefig(params["output_figure"])
    if "output_data" in params:
        np.savetxt(params["output_data"], rta, header="PHI,THETA,FREQUENCY,WAVELENGTH,R,T,A", delimiter=",", comments='')

"""
    Generate S-matrices
"""


from bast.scattering import scattering_interface
import matplotlib.pyplot as plt

def cuboid_data(o, size=(1,1,1)):
    # Code stolen from
    # https://stackoverflow.com/a/35978146/4124317
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1], o[1], o[1]],          
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]   
    z = [[o[2], o[2], o[2], o[2], o[2]],                       
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],   
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],               
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]               
    return np.array(x), np.array(y), np.array(z)

def data_for_cylinder_along_z(center_x,center_y, center_z, radius,height_z):
    z = center_z + np.linspace(0, height_z, 2)
    theta = np.linspace(0, 2*np.pi, 20)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    plt.plot(center_x+radius*np.cos(theta), center_y+radius*np.sin(theta), center_z, 'k-')
    plt.plot(center_x+radius*np.cos(theta), center_y+radius*np.sin(theta), height_z+center_z, 'k-')
    for t in theta:
        ex = center_x+radius*np.cos(t)
        ey = center_y+radius*np.sin(t)
        plt.plot([ex, ex], [ey, ey], [center_z+0, center_z+height_z], 'k-', alpha=0.4)
    return x_grid,y_grid,z_grid

def plotCubeAt(pos=(0,0,0), size=(1,1,1), ax=None,**kwargs):
    # Plotting a cube element at position pos
    if ax !=None:
        X, Y, Z = cuboid_data( pos, size )
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, **kwargs, zorder=0.4, linewidth=4)
        
        mx, Mx, my, My, mz, Mz = np.min(X), np.max(X), np.min(Y), np.max(Y), np.min(Z), np.max(Z)
        for e2 in [mz, Mz]:
            for ee in [My, my]:
                plt.plot([mx, Mx], [ee, ee], [e2, e2], "k", lw=1,alpha=0.8)
        for e2 in [mx, Mx]:
            for ee in [My, my]:
                plt.plot([e2, e2], [ee, ee], [mz, Mz], "k", lw=1,alpha=0.8)
        for e2 in [mx, Mx]:
            for ee in [Mz, mz]:
                plt.plot([e2, e2], [my, My], [ee, ee], "k", lw=1,alpha=0.8)

def plot_layer(ax, layer, depth=0.0):
    if layer in config.stacking:
        if "repeat" in config.stacking[layer]:
            for i in range(config.stacking[layer]["repeat"]):
                for l in config.stacking[layer]["layers"]:
                    depth = plot_layer(ax, l, depth=depth)
        else:
            for l in config.stacking[layer]["layers"]:
                depth = plot_layer(ax, l, depth=depth)
    elif layer in config.layers:
        ll = config.layers[layer]
        posz = depth
        depth = depth + ll["depth"]
        if not "geometry" in ll.keys():
            ll["geometry"] = []
        for geo in ll["geometry"]:
            if geo["shape"] == "disc":
                center = geo["center"]
                radius = geo["radius"]
                geo_color = config.materials[geo["material"]]["color"]

                
                Xc,Yc,Zc = data_for_cylinder_along_z(*center, posz, radius, ll["depth"])
                ax.plot_surface(Xc, Yc, Zc, zorder=0.4, color=geo_color)
            if geo["shape"] == "rectangle":
                geo_color = config.materials[geo["material"]]["color"]
                center = geo["center"]
                size = geo["size"]
                plotCubeAt(pos=(*center, posz), size=(*size, ll["depth"]), ax=ax, color=geo_color, alpha=0.2)

        layer_color = config.materials[ll["material"]]["color"]
        plotCubeAt(pos=(0,0, posz), size=(1,1, ll["depth"]), ax=ax, color=layer_color, alpha=0.2)
    return depth

        

def view_structure(device):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d',computed_zorder=False, proj_type="ortho")
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    
    zmax = plot_layer(ax, device)
    ax.set_zlim(0, zmax)

    plt.show()



study = config.studies[args.study]
if study["type"] == "prefab_view_structure":
    view_structure(study["device"])
elif study["type"] == "prefab_spectrum":
    materials = load_materials(config.materials)
    layers = load_layers(None)
    compute_spectrum(layers, materials, study["parameters"], study["device"])