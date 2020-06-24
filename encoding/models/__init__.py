from .model_zoo import get_model
from .model_store import get_model_file
from .base import *
from .encnet import *

from .fcn import *
from .psp import *
from .deeplabv3 import *
from .deeplabv3plus import*

from .up_fcn import *
from .up_psp import *
from .up_deeplabv3 import *

def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn,
        'encnet': get_encnet,

        'deeplabv3plus': get_deeplabv3plus,
        'deeplabv3': get_deeplabv3,
        'psp': get_psp,
        'up_fcn': get_up_fcn,
        'up_psp': get_up_psp,
        'up_deeplabv3': get_up_deeplabv3,
    }
    return models[name.lower()](**kwargs)
