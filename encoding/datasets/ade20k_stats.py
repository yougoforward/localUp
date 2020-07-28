###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################

import os
import numpy as np

import torch

from PIL import Image
from base import BaseDataset

class ADE20KSegmentation(BaseDataset):
    BASE_DIR = 'ADEChallengeData2016'
    NUM_CLASS = 150
    def __init__(self, root=os.path.expanduser('../../datasets/ade20k'), split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(ADE20KSegmentation, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        # assert exists and prepare dataset automatically
        root = os.path.join(root, self.BASE_DIR)
        assert os.path.exists(root), "Please setup the dataset using" + \
            "encoding/scripts/prepare_ade20k.py"
        self.images, self.masks = _get_ade20k_pairs(root, split)
        if split != 'test':
            assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + root + "\n"))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
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
        # # general resize, normalize and toTensor
        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     mask = self.target_transform(mask)
        return img, mask

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int64') - 1
        return torch.from_numpy(target)

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 1


def _get_ade20k_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for filename in os.listdir(img_folder):
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".jpg"):
                imgpath = os.path.join(img_folder, filename)
                maskname = basename + '.png'
                maskpath = os.path.join(mask_folder, maskname)
                if os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    print('cannot find the mask:', maskpath)
        return img_paths, mask_paths

    if split == 'train':
        img_folder = os.path.join(folder, 'images/training')
        mask_folder = os.path.join(folder, 'annotations/training')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        print('len(img_paths):', len(img_paths))
        assert len(img_paths) == 20210
    elif split == 'val':
        img_folder = os.path.join(folder, 'images/validation')
        mask_folder = os.path.join(folder, 'annotations/validation')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        assert len(img_paths) == 2000
    elif split == 'test':
        folder = os.path.join(folder, '../release_test')
        with open(os.path.join(folder, 'list.txt')) as f:
            img_paths = [os.path.join(folder, 'testing', line.strip()) for line in f]
        assert len(img_paths) == 3352
        return img_paths, None
    else:
        assert split == 'trainval'
        train_img_folder = os.path.join(folder, 'images/training')
        train_mask_folder = os.path.join(folder, 'annotations/training')
        val_img_folder = os.path.join(folder, 'images/validation')
        val_mask_folder = os.path.join(folder, 'annotations/validation')
        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
        assert len(img_paths) == 22210
    return img_paths, mask_paths



trainset = ADE20KSegmentation(split='train', mode='train')

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

norm_tvect = torch.tensor([0.0864, 0.1547, 0.1057, 0.0870, 0.0609, 0.0476, 0.0444, 0.0392, 0.0225,
                            0.0196, 0.0178, 0.0178, 0.0164, 0.0156, 0.0147, 0.0116, 0.0108, 0.0107,
                            0.0099, 0.0102, 0.0101, 0.0098, 0.0072, 0.0067, 0.0065, 0.0061, 0.0060,
                            0.0054, 0.0052, 0.0044, 0.0044, 0.0044, 0.0044, 0.0032, 0.0031, 0.0029,
                            0.0029, 0.0025, 0.0024, 0.0023, 0.0023, 0.0023, 0.0021, 0.0021, 0.0020,
                            0.0019, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0017,
                            0.0016, 0.0016, 0.0017, 0.0017, 0.0016, 0.0015, 0.0015, 0.0015, 0.0015,
                            0.0015, 0.0013, 0.0014, 0.0014, 0.0014, 0.0013, 0.0013, 0.0013, 0.0012,
                            0.0011, 0.0012, 0.0012, 0.0010, 0.0010, 0.0010, 0.0008, 0.0009, 0.0009,
                            0.0009, 0.0008, 0.0008, 0.0008, 0.0008, 0.0007, 0.0007, 0.0007, 0.0007,
                            0.0007, 0.0007, 0.0007, 0.0007, 0.0006, 0.0007, 0.0006, 0.0006, 0.0006,
                            0.0006, 0.0006, 0.0006, 0.0006, 0.0006, 0.0005, 0.0005, 0.0005, 0.0005,
                            0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0004,
                            0.0005, 0.0005, 0.0005, 0.0005, 0.0004, 0.0004, 0.0004, 0.0004, 0.0003,
                            0.0004, 0.0003, 0.0004, 0.0003, 0.0004, 0.0003, 0.0003, 0.0004, 0.0003,
                            0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
                            0.0003, 0.0003, 0.0003, 0.0002, 0.0002, 0.0002])
class_balance_weight = torch.tensor([ 0.0772,  0.0431,  0.0631,  0.0766,  0.1095,  0.1399,  0.1502,  0.1702,
                                     0.2958,  0.3400,  0.3738,  0.3749,  0.4059,  0.4266,  0.4524,  0.5725,
                                     0.6145,  0.6240,  0.6709,  0.6517,  0.6591,  0.6818,  0.9203,  0.9965,
                                     1.0272,  1.0967,  1.1202,  1.2354,  1.2900,  1.5038,  1.5160,  1.5172,
                                     1.5036,  2.0746,  2.1426,  2.3159,  2.2792,  2.6468,  2.8038,  2.8777,
                                     2.9525,  2.9051,  3.1050,  3.1785,  3.3533,  3.5300,  3.6120,  3.7006,
                                     3.6790,  3.8057,  3.7604,  3.8043,  3.6610,  3.8268,  4.0644,  4.2698,
                                     4.0163,  4.0272,  4.1626,  4.3702,  4.3144,  4.3612,  4.4389,  4.5612,
                                     5.1537,  4.7653,  4.8421,  4.6813,  5.1037,  5.0729,  5.2657,  5.6153,
                                     5.8240,  5.5360,  5.6373,  6.6972,  6.4561,  6.9555,  7.9239,  7.3265,
                                     7.7501,  7.7900,  8.0528,  8.5415,  8.1316,  8.6557,  9.0550,  9.0081,
                                     9.3262,  9.1391,  9.7237,  9.3775,  9.4592,  9.7883, 10.6705, 10.2113,
                                    10.5845, 10.9667, 10.8754, 10.8274, 11.6427, 11.0687, 10.8417, 11.0287,
                                    12.2030, 12.8830, 12.5082, 13.0703, 13.8410, 12.3264, 12.9048, 12.9664,
                                    12.3523, 13.9830, 13.8105, 14.0345, 15.0054, 13.9801, 14.1048, 13.9025,
                                    13.6179, 17.0577, 15.8351, 17.7102, 17.3153, 19.4640, 17.7629, 19.9093,
                                    16.9529, 19.3016, 17.6671, 19.4525, 20.0794, 18.3574, 19.1219, 19.5089,
                                    19.2417, 20.2534, 20.0332, 21.7496, 21.5427, 20.3008, 21.1942, 22.7051,
                                    23.3359, 22.4300, 20.9934, 26.9073, 31.7362, 30.0784])
