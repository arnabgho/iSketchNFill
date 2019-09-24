from .numpy import *

try:
    import torch
    #import ui_shadow_draw.thinplate.pytorch as torch
except ImportError:
    pass

__version__ = '1.0.0'
