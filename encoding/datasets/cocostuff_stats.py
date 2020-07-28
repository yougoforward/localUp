###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################

import os
import sys
import numpy as np
import random
import math
from PIL import Image, ImageOps, ImageFilter

import torch
import torch.utils.data as data
import torchvision.transforms as transform
import re
from tqdm import tqdm
from base import BaseDataset


class CocostuffSegmentation(BaseDataset):
    BASE_DIR = 'cocostuff'
    NUM_CLASS = 171

    def __init__(self, root='../../datasets', split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(CocostuffSegmentation, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        # assert exists
        root = os.path.join(root, self.BASE_DIR)
        assert os.path.exists(root), "Please download the dataset!!"

        self.images, self.masks = _get_cocostuff_pairs(root, split)
        if split != 'vis':
            assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: \
                " + root + "\n"))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'vis':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])

        mask = Image.open(self.masks[index])

        # # synchrosized transform
        # if self.mode == 'train':
        #     img, mask = self._sync_transform(img, mask)
        # elif self.mode == 'val':
        #     img, mask = self._val_sync_transform(img, mask)
        # else:
        #     assert self.mode == 'testval'
        #     mask = self._mask_transform(mask)
        #
        # # general resize, normalize and toTensor
        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     mask = self.target_transform(mask)

        return img, mask

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')-1
        # target[target == 255] = -1
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0


def _get_cocostuff_pairs(folder, split='train'):
    def get_path_pairs(folder, split_f):
        img_paths = []
        mask_paths = []
        with open(split_f, 'r') as lines:
            for line in tqdm(lines):
                ll_str = re.split(' ', line)
                imgpath = os.path.join(folder, ll_str[0].lstrip('/').lstrip(

                ).rstrip())
                maskpath = os.path.join(folder, ll_str[1].lstrip('/').rstrip())
                if os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    print('cannot find the mask:', maskpath)
        return img_paths, mask_paths

    if split == 'train':
        split_f = os.path.join(folder, 'train.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)
    elif split == 'val':
        split_f = os.path.join(folder, 'val.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)
    elif split == 'test':
        split_f = os.path.join(folder, 'test.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)
    else:
        split_f = os.path.join(folder, 'all.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)

    return img_paths, mask_paths


trainset = CocostuffSegmentation(split='train', mode='train')

print(len(trainset.images))
nclass = trainset.NUM_CLASS
tvect = torch.zeros(nclass)
for index in range(len(trainset.images)):
    print(index)
    img, mask = trainset.__getitem__(index)
    hist = torch.histc(torch.tensor(np.array(mask)).float(), bins=nclass, min=0, max=nclass - 1)
    tvect = tvect+hist

norm_tvect = tvect/torch.sum(tvect)
print(norm_tvect)

# nclass = trainset.NUM_CLASS
# tvect = torch.zeros(nclass)
# all = torch.zeros(1)
# norm_tvect = torch.zeros(nclass)
# for index in range(len(trainset.images)):
#     print(index)
#     img, mask = trainset.__getitem__(index)
#     hist = torch.histc(torch.tensor(np.array(mask)).float(), bins=nclass, min=0, max=nclass - 1)
#     # tvect = tvect+hist
#     tvect = norm_tvect*all + hist
#     all = torch.sum(tvect)
#     norm_tvect = tvect/all
# print(norm_tvect)
class_balance_weight = 1/nclass/norm_tvect
print(class_balance_weight)

norm_tvect = torch.tensor([ 1.2043e-01, 9.0373e-02, 1.4897e-03, 6.1689e-03, 4.6037e-03, 4.1325e-03,
                            7.3344e-03, 6.8878e-03, 5.4393e-03, 2.4365e-03, 6.5168e-04, 1.1026e-03,
                            9.6805e-04, 6.2689e-04, 3.6659e-03, 1.6204e-03, 5.9206e-03, 4.7471e-03,
                            3.0468e-03, 2.1170e-03, 2.8460e-03, 4.8498e-03, 1.6055e-03, 2.8130e-03,
                            3.2761e-03, 8.7104e-04, 2.7435e-03, 8.3766e-04, 4.6125e-05, 2.8727e-03,
                            2.5500e-04, 2.5209e-04, 3.3534e-04, 1.4162e-04, 7.4327e-04, 1.3461e-04,
                            8.6582e-05, 5.8098e-04, 8.6610e-04, 4.4391e-04, 1.7457e-03, 8.3857e-04,
                            2.7223e-03, 3.0100e-04, 4.4168e-04, 3.0429e-04, 7.6109e-03, 2.2457e-03,
                            9.4593e-04, 4.1564e-03, 1.3269e-03, 1.0119e-03, 5.6660e-04, 1.1714e-03,
                            8.6754e-03, 1.8533e-03, 3.6937e-03, 6.5205e-03, 5.0043e-03, 2.2079e-03,
                            8.9481e-03, 2.4571e-02, 2.9934e-03, 3.6868e-03, 3.2860e-03, 2.1354e-04,
                            4.7088e-04, 1.4885e-03, 7.4122e-04, 7.7214e-04, 2.9598e-03, 7.1350e-05,
                            1.5839e-03, 2.9183e-03, 2.1395e-03, 1.0688e-03, 1.4976e-03, 3.0483e-04,
                            2.5774e-03, 2.2414e-05, 1.2346e-04, 2.4526e-03, 7.0195e-04, 3.9360e-04,
                            8.9864e-04, 2.8042e-02, 5.6094e-03, 3.3636e-03, 4.8844e-04, 1.1063e-03,
                            3.3723e-03, 6.7902e-03, 6.2410e-04, 2.0487e-03, 9.7266e-04, 3.3303e-02,
                            3.2260e-03, 4.2426e-03, 4.2689e-03, 2.4646e-03, 9.3092e-03, 4.4651e-03,
                            1.2211e-02, 5.6951e-04, 4.7516e-03, 1.0506e-03, 4.5450e-03, 4.5885e-03,
                            1.2310e-03, 6.9595e-04, 3.1308e-03, 5.5593e-04, 5.2027e-03, 3.9050e-02,
                            2.5265e-03, 1.4172e-02, 2.2582e-03, 4.0229e-03, 1.2831e-03, 7.4837e-04,
                            4.0047e-04, 2.0297e-03, 3.0994e-03, 7.4848e-05, 3.0264e-03, 7.7994e-04,
                            4.6988e-04, 2.6757e-03, 1.8881e-03, 1.4170e-02, 1.1931e-04, 8.2365e-03,
                            7.8989e-04, 2.4997e-03, 2.0022e-02, 2.4209e-03, 1.7608e-03, 2.7455e-03,
                            2.3837e-02, 3.4237e-03, 1.2544e-03, 2.0028e-03, 5.6192e-04, 7.6738e-03,
                            2.1031e-02, 1.5507e-03, 4.9339e-02, 6.6243e-04, 1.8448e-02, 9.1711e-05,
                            8.8562e-04, 1.1208e-03, 1.3414e-03, 9.2876e-03, 5.7294e-03, 4.9852e-04,
                            2.4433e-03, 6.3039e-04, 5.3051e-02, 1.5828e-03, 4.1943e-03, 2.4984e-03,
                            4.8283e-02, 2.0506e-03, 2.0741e-03, 5.8026e-03, 3.5199e-03, 8.9545e-03,
                            3.0665e-04, 2.1475e-03, 9.6232e-03])

class_balance_weight = torch.tensor([   4.8557e-02, 6.4709e-02, 3.9255e+00, 9.4797e-01, 1.2703e+00, 1.4151e+00,
                                        7.9733e-01, 8.4903e-01, 1.0751e+00, 2.4001e+00, 8.9736e+00, 5.3036e+00,
                                        6.0410e+00, 9.3285e+00, 1.5952e+00, 3.6090e+00, 9.8772e-01, 1.2319e+00,
                                        1.9194e+00, 2.7624e+00, 2.0548e+00, 1.2058e+00, 3.6424e+00, 2.0789e+00,
                                        1.7851e+00, 6.7138e+00, 2.1315e+00, 6.9813e+00, 1.2679e+02, 2.0357e+00,
                                        2.2933e+01, 2.3198e+01, 1.7439e+01, 4.1294e+01, 7.8678e+00, 4.3444e+01,
                                        6.7543e+01, 1.0066e+01, 6.7520e+00, 1.3174e+01, 3.3499e+00, 6.9737e+00,
                                        2.1482e+00, 1.9428e+01, 1.3240e+01, 1.9218e+01, 7.6836e-01, 2.6041e+00,
                                        6.1822e+00, 1.4070e+00, 4.4074e+00, 5.7792e+00, 1.0321e+01, 4.9922e+00,
                                        6.7408e-01, 3.1554e+00, 1.5832e+00, 8.9685e-01, 1.1686e+00, 2.6487e+00,
                                        6.5354e-01, 2.3801e-01, 1.9536e+00, 1.5862e+00, 1.7797e+00, 2.7385e+01,
                                        1.2419e+01, 3.9287e+00, 7.8897e+00, 7.5737e+00, 1.9758e+00, 8.1962e+01,
                                        3.6922e+00, 2.0039e+00, 2.7333e+00, 5.4717e+00, 3.9048e+00, 1.9184e+01,
                                        2.2689e+00, 2.6091e+02, 4.7366e+01, 2.3844e+00, 8.3310e+00, 1.4857e+01,
                                        6.5076e+00, 2.0854e-01, 1.0425e+00, 1.7386e+00, 1.1973e+01, 5.2862e+00,
                                        1.7341e+00, 8.6124e-01, 9.3702e+00, 2.8545e+00, 6.0123e+00, 1.7560e-01,
                                        1.8128e+00, 1.3784e+00, 1.3699e+00, 2.3728e+00, 6.2819e-01, 1.3097e+00,
                                        4.7892e-01, 1.0268e+01, 1.2307e+00, 5.5662e+00, 1.2867e+00, 1.2745e+00,
                                        4.7505e+00, 8.4029e+00, 1.8679e+00, 1.0519e+01, 1.1240e+00, 1.4975e-01,
                                        2.3146e+00, 4.1265e-01, 2.5896e+00, 1.4537e+00, 4.5575e+00, 7.8143e+00,
                                        1.4603e+01, 2.8812e+00, 1.8868e+00, 7.8131e+01, 1.9323e+00, 7.4980e+00,
                                        1.2446e+01, 2.1856e+00, 3.0973e+00, 4.1270e-01, 4.9016e+01, 7.1001e-01,
                                        7.4035e+00, 2.3395e+00, 2.9207e-01, 2.4156e+00, 3.3211e+00, 2.1300e+00,
                                        2.4533e-01, 1.7081e+00, 4.6621e+00, 2.9199e+00, 1.0407e+01, 7.6207e-01,
                                        2.7806e-01, 3.7711e+00, 1.1852e-01, 8.8280e+00, 3.1700e-01, 6.3765e+01,
                                        6.6032e+00, 5.2177e+00, 4.3596e+00, 6.2965e-01, 1.0207e+00, 1.1731e+01,
                                        2.3935e+00, 9.2767e+00, 1.1023e-01, 3.6947e+00, 1.3943e+00, 2.3407e+00,
                                        1.2112e-01, 2.8518e+00, 2.8195e+00, 1.0078e+00, 1.6614e+00, 6.5307e-01,
                                        1.9070e+01, 2.7231e+00, 6.0769e-01])
