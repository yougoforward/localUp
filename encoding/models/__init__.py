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
from .up_fcn_3x3_s4_dilation import *
from .up_fcn_3x3_s4_dilation_256 import *


from .up_fcn_3x3_s4 import *
from .up_fcn_3x3_s16 import *

from .up_fcn_3x3_s8 import *
from .up_fcn_5x5_s8 import *
from .up_fcn_5x5_s4 import *
from .up_fcn_com import *
from .up_fcn_dilation import *
from .up_fcn_dilation_v2 import *
from .up_fcn_dilation_v3 import *

from .fcn_fpn import *

from .blur_detect import *
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
        'up_fcn_3x3_s4': get_up_fcn_3x3_s4,
        'up_fcn_3x3_s4_dilation':get_up_fcn_3x3_s4_dilation,
        'up_fcn_3x3_s4_dilation_256':get_up_fcn_3x3_s4_dilation_256,

        'up_fcn_3x3_s8': get_up_fcn_3x3_s8,
        'up_fcn_3x3_s16': get_up_fcn_3x3_s16,
        'up_fcn_5x5_s8': get_up_fcn_5x5_s8,
        'up_fcn_5x5_s4': get_up_fcn_5x5_s4,
        'up_fcn_com':get_up_fcn_com,
        'up_fcn_dilation': get_up_fcn_dilation,
        'up_fcn_dilation_v2': get_up_fcn_dilation_v2,
        'up_fcn_dilation_v3': get_up_fcn_dilation_v3,

        'blur_detect': get_blur_detect,
        'fcn_fpn': get_fcn_fpn,

    }
    return models[name.lower()](**kwargs)
