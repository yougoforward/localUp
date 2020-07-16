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
from .fpn_psp import *
from .up_deeplabv3 import *
from .up_fcn_3x3_s4_dilation import *
from .up_fcn_3x3_s4_dilation_256 import *


from .up_fcn_3x3_s4 import *
from .up_fcn_3x3_s16 import *

from .up_fcn_3x3_s8 import *
from .up_fcn_5x5_s8 import *
from .up_fcn_5x5_s4 import *
from .up_fcn_5x5_s16 import *

from .up_fcn_com import *
from .up_fcn_dilation import *
from .up_fcn_dilation_v2 import *
from .up_fcn_dilation_v3 import *
from .up_fcn_dilation_s8plus import *

from .fcn_fpn import *
from .fcn_fpn2 import *

from .fcn_fpn_2048 import *
from .up_fcn_2048 import *

from .fcn_fpn_s16 import *
from .fcn_fpn_256 import *
from .fcn_fpn_nobn import *
from .fcn_fpn_s4 import *
from .pano_fpn import *
from .fpn_gsnet import *
from .fpn3x3_gsnet import *

from .fpn_enc import *
from .up_enc import *
from .up_gsnet import *
from .gsnet import *
from .jpux_gsf import *
from .jpu_gsf import *
from .jpux_gsf_oc import *
from .fpn2_gsnet import *
from .dfpn_gsf import *
from .dfpn import *
from .dfpn2_gsf import *
from .dfpn2 import *
from .dfpn3 import *
from .dfpn3_gsf import *
from .dfpn4 import *
from .dfpn4_gsf import *
from .dfpn5_gsf import *
from .dfpn6_gsf import *
from .dfpn7_gsf import *
from .dfpn8_gsf import *
from .dfpn9_gsf import *
from .dfpn10_gsf import *

from .blur_detect import *
from .blur_dfpn_gsf import *
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
        'fpn_psp': get_fpn_psp,
        
        'up_deeplabv3': get_up_deeplabv3,
        'up_fcn_3x3_s4': get_up_fcn_3x3_s4,
        'up_fcn_3x3_s4_dilation':get_up_fcn_3x3_s4_dilation,
        'up_fcn_3x3_s4_dilation_256':get_up_fcn_3x3_s4_dilation_256,

        'up_fcn_3x3_s8': get_up_fcn_3x3_s8,
        'up_fcn_3x3_s16': get_up_fcn_3x3_s16,
        'up_fcn_5x5_s8': get_up_fcn_5x5_s8,
        'up_fcn_5x5_s4': get_up_fcn_5x5_s4,
        'up_fcn_5x5_s16': get_up_fcn_5x5_s16,

        'up_fcn_com':get_up_fcn_com,
        'up_fcn_dilation': get_up_fcn_dilation,
        'up_fcn_dilation_v2': get_up_fcn_dilation_v2,
        'up_fcn_dilation_v3': get_up_fcn_dilation_v3,
        'up_fcn_dilation_s8plus': get_up_fcn_dilation_s8plus,
        
        'blur_detect': get_blur_detect,
        'blur_dfpn_gsf': get_blur_dfpn_gsf,
        'fcn_fpn': get_fcn_fpn,
        'fcn_fpn2': get_fcn_fpn2,
        'fcn_fpn_s16': get_fcn_fpn_s16,
        'fcn_fpn_256': get_fcn_fpn_256,
        'fcn_fpn_nobn': get_fcn_fpn_nobn,
        'fcn_fpn_s4': get_fcn_fpn_s4,
        'pano_fpn': get_pano_fpn,

        'fcn_fpn_2048': get_fcn_fpn_2048,
        'up_fcn_2048': get_up_fcn_2048,
        'fpn_gsnet': get_fpn_gsnet,
        'fpn3x3_gsnet': get_fpn3x3_gsnet,

        'fpn_enc': get_fpn_enc,
        'up_enc': get_up_enc,
        'up_gsnet': get_up_gsnet,
        'gsnet': get_gsnetnet,
        'jpux_gsf': get_jpux_gsf,
        'jpu_gsf': get_jpu_gsf,
        'jpux_gsf_oc': get_jpux_gsf_oc,
        'fpn2_gsnet': get_fpn2_gsnet,
        'dfpn_gsf': get_dfpn_gsf,
        'dfpn': get_dfpn,
        'dfpn2_gsf': get_dfpn2_gsf,
        'dfpn2': get_dfpn2,
        'dfpn3_gsf': get_dfpn3_gsf,
        'dfpn3': get_dfpn3,
        'dfpn4_gsf': get_dfpn4_gsf,
        'dfpn4': get_dfpn4,
        'dfpn5_gsf': get_dfpn5_gsf,
        'dfpn6_gsf': get_dfpn6_gsf,
        'dfpn7_gsf': get_dfpn7_gsf,
        'dfpn8_gsf': get_dfpn8_gsf,
        'dfpn9_gsf': get_dfpn9_gsf,
        'dfpn10_gsf': get_dfpn10_gsf,

    }
    return models[name.lower()](**kwargs)
