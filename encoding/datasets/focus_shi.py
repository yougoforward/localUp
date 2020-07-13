import os
import numpy as np

import torch
import random
import numpy as np
import cv2
import torch.utils.data as data
import numpy
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm
import torchvision.transforms as transforms

from .base import BaseDataset

class Blur2Segmentation(BaseDataset):
    CLASSES = [
        'blur', 'clear'
    ]
    NUM_CLASS = 2
    BASE_DIR = 'focus_shi'
    def __init__(self, root=os.path.expanduser('./datasets'), split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(Blur2Segmentation, self).__init__(root, split, mode, transform,
                                              target_transform, **kwargs)
        _voc_root = os.path.join(self.root, self.BASE_DIR)
        _mask_dir = os.path.join(_voc_root, 'bin_label')
        _image_dir = os.path.join(_voc_root, 'image')
        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(_voc_root, 'splits')
        if self.split == 'train':
            _split_f = os.path.join(_splits_dir, 'train_ori.txt')
        elif self.split == 'val':
            _split_f = os.path.join(_splits_dir, 'val.txt')
        elif self.split == 'test':
            _split_f = os.path.join(_splits_dir, 'test.txt')
        else:
            raise RuntimeError('Unknown dataset split.')
        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in tqdm(lines):
                _image = os.path.join(_image_dir, line.rstrip('\n')+".jpg")
                # print(_image)
                assert os.path.isfile(_image)
                self.images.append(_image)
                if self.mode != 'test':
                    _mask = os.path.join(_mask_dir, line.rstrip('\n')+".png")
                    assert os.path.isfile(_mask)
                    self.masks.append(_mask)

        if self.mode != 'test':
            assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        target = Image.open(self.masks[index])
        # synchrosized transform
        if self.mode == 'train':
            img, target = self._sync_transform( img, target)
        elif self.mode == 'val':
            img, target = self._val_sync_transform( img, target)
        else:
            assert self.mode == 'testval'
            target = self._mask_transform(target)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge from 480 to 720)
        short_size = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # colorjitter = transforms.ColorJitter(brightness=0.1, contrast=0.5, saturation=0.5, hue=0.1)
        # img = colorjitter(img)

        # random rotate
        
        img, mask = RandomRotation(img, mask, 45, is_continuous=False)

        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
        # # gaussian blur as in PSP
        # if random.random() < 0.5:
        #     img = img.filter(ImageFilter.GaussianBlur(
        #         radius=random.random()))

        #random hsv
        # img = RandomHSV(img, 10, 10, 10)
        # #random contrast
        # img=RandomContrast(img)
        # #random perm
        # img = RandomPerm(img)
        # final transform
        return img, self._mask_transform(mask)
    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0
        
def RandomHSV(image, h_r, s_r, v_r):
    """Generate randomly the image in hsv space."""
    image = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h = hsv[:,:,0].astype(np.int32)
    s = hsv[:,:,1].astype(np.int32)
    v = hsv[:,:,2].astype(np.int32)
    delta_h = np.random.randint(-h_r, h_r)
    delta_s = np.random.randint(-s_r, s_r)
    delta_v = np.random.randint(-v_r, v_r)
    h = (h + delta_h)%180
    s = s + delta_s
    s[s>255] = 255
    s[s<0] = 0
    v = v + delta_v
    v[v>255] = 255
    v[v<0] = 0
    hsv = np.stack([h,s,v], axis=-1).astype(np.uint8)
    new_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.uint8)
    new_image = Image.fromarray(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    return new_image


def RandomRotation(image, segmentation, angle_r, is_continuous=False):
    """Randomly rotate image"""
    seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST
    image = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR)
    segmentation = numpy.asarray(segmentation)
    row, col, _ = image.shape
    # rand_angle = np.random.randint(-angle_r, angle_r) if angle_r != 0 else 0
    rand_angle = np.random.randint(-4, 4)*angle_r
    m = cv2.getRotationMatrix2D(center=(col/2, row/2), angle=rand_angle, scale=1)

    new_image = cv2.warpAffine(image, m, (col,row), flags=cv2.INTER_CUBIC, borderValue=0)
    new_segmentation = cv2.warpAffine(segmentation, m, (col,row), flags=seg_interpolation, borderValue=255)

    new_image = Image.fromarray(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    new_segmentation = Image.fromarray(new_segmentation)
    return new_image, new_segmentation


def RandomPerm(img, ratio=0.5):
    perms = ((0, 1, 2), (0, 2, 1),
             (1, 0, 2), (1, 2, 0),
             (2, 0, 1), (2, 1, 0))
    if random.random() > ratio:
        return img

    img_mode = img.mode
    swap = perms[random.randint(0, len(perms) - 1)]
    img = np.asarray(img)
    img = img[:, :, swap]
    img = Image.fromarray(img.astype(np.uint8), mode=img_mode)
    return img

def RandomContrast(img, lower=0.5, upper=1.5, ratio=0.5):

    assert upper >= lower, "contrast upper must be >= lower."
    assert lower >= 0, "contrast lower must be non-negative."
    assert isinstance(img, Image.Image)
    if random.random() > ratio:
        return img
    img_mode = img.mode
    img = np.array(img).astype(np.float32)
    img *= random.uniform(lower, upper)
    img = np.clip(img, 0, 255)
    img = Image.fromarray(img.astype(np.uint8), mode=img_mode)
    return img