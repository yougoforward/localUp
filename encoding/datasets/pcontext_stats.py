###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################


import os
import numpy as np

import torch

from PIL import Image
from tqdm import trange

from base import BaseDataset

class ContextSegmentation(BaseDataset):
    BASE_DIR = 'VOCdevkit/VOC2010'
    NUM_CLASS = 60
    def __init__(self, root=os.path.expanduser('../../datasets/pcontext'), split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(ContextSegmentation, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        from detail import Detail
        #from detail import mask
        root = os.path.join(root, self.BASE_DIR)
        annFile = os.path.join(root, 'trainval_merged.json')
        imgDir = os.path.join(root, 'JPEGImages')
        # training mode
        self.detail = Detail(annFile, imgDir, split)
        self.transform = transform
        self.target_transform = target_transform
        self.ids = self.detail.getImgs()
        # generate masks
        self._mapping = np.sort(np.array([
            0, 2, 259, 260, 415, 324, 9, 258, 144, 18, 19, 22,
            23, 397, 25, 284, 158, 159, 416, 33, 162, 420, 454, 295, 296,
            427, 44, 45, 46, 308, 59, 440, 445, 31, 232, 65, 354, 424,
            68, 326, 72, 458, 34, 207, 80, 355, 85, 347, 220, 349, 360,
            98, 187, 104, 105, 366, 189, 368, 113, 115]))
        self._key = np.array(range(len(self._mapping))).astype('uint8')
        mask_file = os.path.join(root, self.split+'.pth')
        print('mask_file:', mask_file)
        if os.path.exists(mask_file):
            self.masks = torch.load(mask_file)
        else:
            self.masks = self._preprocess(mask_file)

    def _class_to_index(self, mask):
        # assert the values
        values = np.unique(mask)
        for i in range(len(values)):
            assert(values[i] in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def _preprocess(self, mask_file):
        masks = {}
        tbar = trange(len(self.ids))
        print("Preprocessing mask, this will take a while." + \
            "But don't worry, it only run once for each split.")
        for i in tbar:
            img_id = self.ids[i]
            mask = Image.fromarray(self._class_to_index(
                self.detail.getMask(img_id)))
            masks[img_id['image_id']] = mask
            tbar.set_description("Preprocessing masks {}".format(img_id['image_id']))
        torch.save(masks, mask_file)
        return masks

    def __getitem__(self, index):
        img_id = self.ids[index]
        path = img_id['file_name']
        iid = img_id['image_id']
        img = Image.open(os.path.join(self.detail.img_folder, path)).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(path)
        # convert mask to 60 categories
        mask = self.masks[iid]
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
        target = np.array(mask).astype('int32') - 1
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.ids)

    @property
    def pred_offset(self):
        return 1


trainset = ContextSegmentation(split='train', mode='train')

print(len(trainset.ids))
nclass = trainset.NUM_CLASS
tvect = torch.zeros(nclass)
for index in range(len(trainset.ids)):
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
# for index in range(len(trainset.ids)):
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

norm_tvect = torch.tensor([ 1.2602e-01, 8.0296e-03, 9.1847e-04, 3.0274e-03, 7.5552e-03, 5.9409e-04,
                            8.1038e-03, 8.9615e-03, 6.9367e-03, 1.7838e-03, 4.6309e-03, 6.0635e-02,
                            1.1724e-02, 6.6703e-03, 2.1573e-02, 3.4000e-02, 5.7338e-03, 1.3664e-02,
                            7.3279e-03, 8.1524e-04, 5.5450e-03, 9.1742e-04, 4.4557e-03, 2.9369e-02,
                            5.1076e-03, 1.1106e-02, 3.0020e-02, 1.6574e-03, 8.7943e-04, 6.2208e-02,
                            6.0651e-02, 9.1029e-03, 8.1588e-04, 1.1271e-03, 1.1180e-02, 8.7201e-03,
                            6.3695e-05, 7.1739e-02, 8.7277e-04, 2.4866e-03, 5.7959e-03, 2.4223e-02,
                            3.7759e-03, 6.5050e-03, 3.7530e-03, 3.8033e-03, 1.2100e-03, 8.4163e-02,
                            4.5113e-03, 1.6030e-02, 8.0904e-03, 2.9863e-03, 1.2586e-02, 5.5393e-02,
                            9.6095e-04, 7.7270e-03, 6.1239e-02, 2.8378e-02, 8.3517e-03, 3.7875e-03])

class_balance_weight = torch.tensor([1.3225e-01, 2.0757e+00, 1.8146e+01, 5.5052e+00, 2.2060e+00, 2.8054e+01,
                                     2.0566e+00, 1.8598e+00, 2.4027e+00, 9.3435e+00, 3.5990e+00, 2.7487e-01,
                                     1.4216e+00, 2.4986e+00, 7.7258e-01, 4.9020e-01, 2.9067e+00, 1.2197e+00,
                                     2.2744e+00, 2.0444e+01, 3.0057e+00, 1.8167e+01, 3.7405e+00, 5.6749e-01,
                                     3.2631e+00, 1.5007e+00, 5.5519e-01, 1.0056e+01, 1.8952e+01, 2.6792e-01,
                                     2.7479e-01, 1.8309e+00, 2.0428e+01, 1.4788e+01, 1.4908e+00, 1.9113e+00,
                                     2.6166e+02, 2.3233e-01, 1.9096e+01, 6.7025e+00, 2.8756e+00, 6.8804e-01,
                                     4.4140e+00, 2.5621e+00, 4.4409e+00, 4.3821e+00, 1.3774e+01, 1.9803e-01,
                                     3.6944e+00, 1.0397e+00, 2.0601e+00, 5.5811e+00, 1.3242e+00, 3.0088e-01,
                                     1.7344e+01, 2.1569e+00, 2.7216e-01, 5.8731e-01, 1.9956e+00, 4.4004e+00])