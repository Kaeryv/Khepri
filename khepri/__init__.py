import os
if "BAST_MT_ON" in os.environ:
    BAST_MT_ON=bool(int(os.environ["BAST_MT_ON"]))
else:
    BAST_MT_ON=True

from .layer import Layer
from .crystal import Crystal
from .expansion import Expansion
from .draw import Drawing

import pickle
def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def save(path, obj):
    with open(path, "wb") as f:
        return pickle.dump(obj, f)
    