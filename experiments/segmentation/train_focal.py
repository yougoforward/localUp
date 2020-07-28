###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################

import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.deterministic=True

import encoding.utils as utils
from encoding.nn import Focal_SegmentationLosses as SegmentationLosses
from encoding.nn import SyncBatchNorm
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import get_segmentation_dataset
from encoding.models import get_segmentation_model

from .option import Options

torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable



class Trainer():
    def __init__(self, args):
        self.args = args
        # data transforms
        input_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize([.485, .456, .406], [.229, .224, .225])])
        # dataset
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
                       'crop_size': args.crop_size}
        trainset = get_segmentation_dataset(args.dataset, split=args.train_split, mode='train',
                                           **data_kwargs)
        testset = get_segmentation_dataset(args.dataset, split='val', mode ='val',
                                           **data_kwargs)
        # dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True} \
            if args.cuda else {}
        self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size,
                                           drop_last=False, shuffle=True, **kwargs)
        self.valloader = data.DataLoader(testset, batch_size=args.batch_size,
                                         drop_last=False, shuffle=False, **kwargs)
        self.nclass = trainset.num_class
        # model
        model = get_segmentation_model(args.model, dataset = args.dataset,
                                       backbone = args.backbone, dilated = args.dilated, multi_grid =args.multi_grid,
                                       stride =args.stride,
                                       lateral = args.lateral, jpu = args.jpu, aux = args.aux,
                                       se_loss = args.se_loss, norm_layer = SyncBatchNorm,
                                       base_size = args.base_size, crop_size = args.crop_size)
        # print(model)
        # optimizer using different LR
        params_list = [{'params': model.pretrained.parameters(), 'lr': args.lr},]
        if hasattr(model, 'jpu') and model.jpu:
            params_list.append({'params': model.jpu.parameters(), 'lr': args.lr*10})
        if hasattr(model, 'head'):
            params_list.append({'params': model.head.parameters(), 'lr': args.lr*10})
        if hasattr(model, 'auxlayer'):
            params_list.append({'params': model.auxlayer.parameters(), 'lr': args.lr*10})
        optimizer = torch.optim.SGD(params_list, lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay)

        class_balance_weight = 'None'
        if args.dataset=="pcontext60":
            class_balance_weight = torch.tensor([1.3225e-01, 2.0757e+00, 1.8146e+01, 5.5052e+00, 2.2060e+00, 2.8054e+01,
                                                 2.0566e+00, 1.8598e+00, 2.4027e+00, 9.3435e+00, 3.5990e+00, 2.7487e-01,
                                                 1.4216e+00, 2.4986e+00, 7.7258e-01, 4.9020e-01, 2.9067e+00, 1.2197e+00,
                                                 2.2744e+00, 2.0444e+01, 3.0057e+00, 1.8167e+01, 3.7405e+00, 5.6749e-01,
                                                 3.2631e+00, 1.5007e+00, 5.5519e-01, 1.0056e+01, 1.8952e+01, 2.6792e-01,
                                                 2.7479e-01, 1.8309e+00, 2.0428e+01, 1.4788e+01, 1.4908e+00, 1.9113e+00,
                                                 2.6166e+02, 2.3233e-01, 1.9096e+01, 6.7025e+00, 2.8756e+00, 6.8804e-01,
                                                 4.4140e+00, 2.5621e+00, 4.4409e+00, 4.3821e+00, 1.3774e+01, 1.9803e-01,
                                                 3.6944e+00, 1.0397e+00, 2.0601e+00, 5.5811e+00, 1.3242e+00, 3.0088e-01,
                                                 1.7344e+01, 2.1569e+00, 2.7216e-01, 5.8731e-01, 1.9956e+00,
                                                 4.4004e+00])

        elif args.dataset=="ade20k":
            class_balance_weight = torch.tensor([0.0772, 0.0431, 0.0631, 0.0766, 0.1095, 0.1399, 0.1502, 0.1702,
                                                 0.2958, 0.3400, 0.3738, 0.3749, 0.4059, 0.4266, 0.4524, 0.5725,
                                                 0.6145, 0.6240, 0.6709, 0.6517, 0.6591, 0.6818, 0.9203, 0.9965,
                                                 1.0272, 1.0967, 1.1202, 1.2354, 1.2900, 1.5038, 1.5160, 1.5172,
                                                 1.5036, 2.0746, 2.1426, 2.3159, 2.2792, 2.6468, 2.8038, 2.8777,
                                                 2.9525, 2.9051, 3.1050, 3.1785, 3.3533, 3.5300, 3.6120, 3.7006,
                                                 3.6790, 3.8057, 3.7604, 3.8043, 3.6610, 3.8268, 4.0644, 4.2698,
                                                 4.0163, 4.0272, 4.1626, 4.3702, 4.3144, 4.3612, 4.4389, 4.5612,
                                                 5.1537, 4.7653, 4.8421, 4.6813, 5.1037, 5.0729, 5.2657, 5.6153,
                                                 5.8240, 5.5360, 5.6373, 6.6972, 6.4561, 6.9555, 7.9239, 7.3265,
                                                 7.7501, 7.7900, 8.0528, 8.5415, 8.1316, 8.6557, 9.0550, 9.0081,
                                                 9.3262, 9.1391, 9.7237, 9.3775, 9.4592, 9.7883, 10.6705, 10.2113,
                                                 10.5845, 10.9667, 10.8754, 10.8274, 11.6427, 11.0687, 10.8417, 11.0287,
                                                 12.2030, 12.8830, 12.5082, 13.0703, 13.8410, 12.3264, 12.9048, 12.9664,
                                                 12.3523, 13.9830, 13.8105, 14.0345, 15.0054, 13.9801, 14.1048, 13.9025,
                                                 13.6179, 17.0577, 15.8351, 17.7102, 17.3153, 19.4640, 17.7629, 19.9093,
                                                 16.9529, 19.3016, 17.6671, 19.4525, 20.0794, 18.3574, 19.1219, 19.5089,
                                                 19.2417, 20.2534, 20.0332, 21.7496, 21.5427, 20.3008, 21.1942, 22.7051,
                                                 23.3359, 22.4300, 20.9934, 26.9073, 31.7362, 30.0784])
        elif args.dataset == "cocostuff":
            class_balance_weight = torch.tensor([4.8557e-02, 6.4709e-02, 3.9255e+00, 9.4797e-01, 1.2703e+00, 1.4151e+00,
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



        # criterions
        self.criterion = SegmentationLosses(se_loss=args.se_loss, aux=args.aux,
                                            nclass=self.nclass,
                                            se_weight=args.se_weight,
                                            aux_weight=args.aux_weight,
                                            weight=class_balance_weight)
        self.model, self.optimizer = model, optimizer
        # using cuda
        if args.cuda:
            self.model = DataParallelModel(self.model).cuda()
            self.criterion = DataParallelCriterion(self.criterion).cuda()
        # resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        # clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0
        # lr scheduler
        self.scheduler = utils.LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.trainloader))

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.trainloader)
        for i, (image, target) in enumerate(tbar):
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            if torch_ver == "0.3":
                image = Variable(image)
                target = Variable(target)
            outputs = self.model(image)
            loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, self.args, is_best, filename='checkpoint_{}.pth.tar'.format(epoch))


    def validation(self, epoch):
        # Fast test during the training
        def eval_batch(model, image, target):
            outputs = model(image)
            outputs = gather(outputs, 0, dim=0)
            pred = outputs[0]
            target = target.cuda()
            correct, labeled = utils.batch_pix_accuracy(pred.data, target)
            inter, union = utils.batch_intersection_union(pred.data, target, self.nclass)
            return correct, labeled, inter, union

        is_best = False
        self.model.eval()
        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        tbar = tqdm(self.valloader, desc='\r')
        for i, (image, target) in enumerate(tbar):
            if torch_ver == "0.3":
                image = Variable(image, volatile=True)
                correct, labeled, inter, union = eval_batch(self.model, image, target)
            else:
                with torch.no_grad():
                    correct, labeled, inter, union = eval_batch(self.model, image, target)

            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()
            tbar.set_description(
                'pixAcc: %.3f, mIoU: %.3f' % (pixAcc, mIoU))

        new_pred = (pixAcc + mIoU)/2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': new_pred,
        }, self.args, is_best)


if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val:
            if epoch%20==0 or epoch > trainer.args.epochs-8:
                trainer.validation(epoch)
