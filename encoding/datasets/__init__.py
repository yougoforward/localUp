from .base import *
from .coco import COCOSegmentation
from .ade20k import ADE20KSegmentation
from .pascal_voc import VOCSegmentation
from .pascal_aug import VOCAugSegmentation
from .pcontext import ContextSegmentation
from .cityscapes import CitySegmentation
from .pcontext60 import ContextSegmentation60
from .cocostuff import CocostuffSegmentation
from .focus_shi import Blur2Segmentation
from .sea_707 import SeaSegmentation

datasets = {
    'coco': COCOSegmentation,
    'ade20k': ADE20KSegmentation,
    'pascal_voc': VOCSegmentation,
    'pascal_aug': VOCAugSegmentation,
    'pcontext': ContextSegmentation,
    'pcontext60': ContextSegmentation60,
    'citys': CitySegmentation,
    'cocostuff': CocostuffSegmentation,
    'focus_shi': Blur2Segmentation,
    'sea' : SeaSegmentation
}

def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
