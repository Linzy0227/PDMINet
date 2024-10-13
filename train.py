
#coding=utf-8
import argparse
import os
import time
import logging
import random
import numpy as np
from collections import OrderedDict

import setproctitle
import torch
import torch.optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.transforms import *
# from data.datasets_nii import Brats_loadall_nii, Brats_loadall_test_nii
from data.BraTS import BraTS
from data.data_utils import init_fn
from modules.fusionseg import Model
from utils import Parser, criterions
from utils.parser import setup 
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader 
# from predict import AverageMeter, test_softmax

local_time = time.strftime("%Y%m%d %H%M%S", time.localtime())
parser = argparse.ArgumentParser()

parser.add_argument('--user', default='lizy', type=str)
# setting
parser.add_argument('-batch_size', '--batch_size', default=2, type=int, help='Batch size')
parser.add_argument('--project_root', default='/home/l/data_2/lzy/Bra18/EATE-main', type=str)
# parser.add_argument('--datapath', default='/home/a/data/lzy/Dataset/BRATS2018_Training_none_npy', type=str)
parser.add_argument('--root', default='/home/l/data_1/lzy/BraTSDataset/BraTS2018/MICCAI_BraTS_2018_Data_Training', type=str)
parser.add_argument('--train_file', default='/home/l/data_1/lzy/BraTSDataset/BraTS2018/train.txt', type=str)
parser.add_argument('--dataname', default='BRATS2018', type=str)
parser.add_argument('--savepath', default='/home/l/data_2/lzy/Bra18/EATE-main/output', type=str)
parser.add_argument('--resume', default='/home/l/data_2/lzy/Bra18/EATE-main/output/model_1130.pth', type=str)
parser.add_argument('--num_epochs', default=2000, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--load', default=False, type=bool)  # 加载模型
parser.add_argument('--num_cls', default=4, type=int)


parser.add_argument('--date', default=local_time.split(' ')[0], type=str)
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--iter_per_epoch', default=150, type=int)
parser.add_argument('--region_fusion_start_epoch', default=0, type=int)
parser.add_argument('--seed', default=1024, type=int)
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, 'training')
args.train_transforms = 'Compose([RandCrop3D((128,128,128)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

###tensorboard writer
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))


def main():
    ##########setting gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    ##########setting seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    ##########setting models

    model = Model(num_cls=args.num_cls)
    # print (model)
    model = torch.nn.DataParallel(model).cuda()

    ##########Setting learning schedule and optimizer
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}]
    optimizer = torch.optim.Adam(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)

    logging.info(str(args))
    if os.path.isfile(args.resume) and args.load:
        logging.info('loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info('successfully loading checkpoint {} and traing from epoch:{}'.format(args.resume, args.start_epoch))

    else:
        logging.info('re-traing!')
    # train_set = Brats_loadall_nii(transforms=args.train_transforms, root=args.datapath, num_cls=args.num_cls)
    train_set = BraTS(args.train_file, args.root, mode='train')
    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn)

    ##########Training
    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training############')
    # iter_per_epoch = args.iter_per_epoch
    iter_per_epoch = len(train_loader)
    train_iter = iter(train_loader)
    for epoch in range(args.start_epoch, args.num_epochs):
        setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch + 1, args.num_epochs))
        step_lr = lr_schedule(optimizer, epoch)
        writer.add_scalar('lr', step_lr, global_step=(epoch+1))
        b = time.time()
        for i in range(iter_per_epoch):
            step = (i+1) + epoch*iter_per_epoch
            ###Data load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)
            x, target = data[:2]  # data的前2个数据
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # model.module.is_training = True

            final_out, out1, outs2, outs3, masks1, masks2 = model(x)

            ###Loss compute
            """final loss"""
            final_cross_loss = criterions.softmax_weighted_loss(final_out, target, num_cls=args.num_cls)
            final_dice_loss = criterions.dice_loss(final_out, target, num_cls=args.num_cls)
            final_loss = final_cross_loss + final_dice_loss

            '''out1'''
            out1_cross_loss = criterions.softmax_weighted_loss(out1, target, num_cls=args.num_cls)
            out1_dice_loss = criterions.dice_loss(out1, target, num_cls=args.num_cls)
            out1_loss = out1_cross_loss + out1_dice_loss

            '''outs2'''
            """masks loss"""
            out2_cross_loss = torch.zeros(1).cuda().float()
            out2_dice_loss = torch.zeros(1).cuda().float()
            for out2 in outs2:
                out2_cross_loss += criterions.softmax_weighted_loss(out2, target, num_cls=args.num_cls)
                out2_dice_loss += criterions.dice_loss(out2, target, num_cls=args.num_cls)
            out2_loss = out2_cross_loss + out2_dice_loss

            """outs3 loss"""
            out3_cross_loss = torch.zeros(1).cuda().float()
            out3_dice_loss = torch.zeros(1).cuda().float()
            for out3 in outs3:
                out3_cross_loss += criterions.softmax_weighted_loss(out3, target, num_cls=args.num_cls)
                out3_dice_loss += criterions.dice_loss(out3, target, num_cls=args.num_cls)
            out3_loss = out3_cross_loss + out3_dice_loss

            """mask1"""
            mask1_cross_loss = torch.zeros(1).cuda().float()
            mask1_dice_loss = torch.zeros(1).cuda().float()
            for mask1 in masks1:
                mask1_cross_loss += criterions.softmax_weighted_loss(mask1, target, num_cls=args.num_cls)
                mask1_dice_loss += criterions.dice_loss(mask1, target, num_cls=args.num_cls)
            mask1_loss = mask1_cross_loss + mask1_dice_loss

            """fusion_preds loss"""
            mask2_cross_loss = torch.zeros(1).cuda().float()
            mask2_dice_loss = torch.zeros(1).cuda().float()
            for mask2 in masks2:
                mask2_cross_loss += criterions.softmax_weighted_loss(mask2, target, num_cls=args.num_cls)
                mask2_dice_loss += criterions.dice_loss(mask2, target, num_cls=args.num_cls)
            mask2_loss = mask2_cross_loss + mask2_dice_loss

            """total loss"""
            loss = final_loss + out1_loss + out2_loss + out3_loss + mask1_loss + mask2_loss


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ###log

            writer.add_scalar('loss', loss.item(), global_step=step)
            writer.add_scalar('final_cross_loss', final_cross_loss.item(), global_step=step)
            writer.add_scalar('final_dice_loss', final_dice_loss.item(), global_step=step)
            writer.add_scalar('out1_dice_loss', out1_dice_loss.item(), global_step=step)
            writer.add_scalar('out2_dice_loss', out2_dice_loss.item(), global_step=step)
            writer.add_scalar('out3_dice_loss', out3_dice_loss.item(), global_step=step)
            writer.add_scalar('mask1_dice_loss', mask1_dice_loss.item(), global_step=step)
            writer.add_scalar('mask2_dice_loss', mask2_dice_loss.item(), global_step=step)

            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch + 1), args.num_epochs, (i + 1), iter_per_epoch,
                                                                    loss.item())
            msg += 'finaldice:{:.4f}, cross:{:.4f},'.format(final_dice_loss.item(), final_cross_loss.item())
            msg += 'out1:{:.4f},:{:.4f},'.format(out1_dice_loss.item(), out1_cross_loss.item())
            msg += 'out2:{:.4f},:{:.4f},'.format(out2_dice_loss.item(), out2_cross_loss.item())
            msg += 'out3:{:.4f},:{:.4f},'.format(out3_dice_loss.item(), out3_cross_loss.item())
            msg += 'mask1:{:.4f},:{:.4f},'.format(mask1_dice_loss.item(), mask1_cross_loss.item())
            msg += 'mask2:{:.4f},:{:.4f}'.format(mask2_dice_loss.item(), mask2_cross_loss.item())
            logging.info(msg)
        logging.info('train time per epoch: {}'.format(time.time() - b))

        ##########model save
        file_name = os.path.join(ckpts, 'model_last.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            },
            file_name)
        
        if (epoch+1) % 10 == 0 or (epoch>=(args.num_epochs-10)):
            file_name = os.path.join(ckpts, 'model_{}.pth'.format(epoch+1))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                },
                file_name)

    msg = 'total time: {:.4f} hours'.format((time.time() - start)/3600)
    logging.info(msg)


if __name__ == '__main__':
    main()

# python train.py
