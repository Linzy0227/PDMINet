import os
import torch
from torch.utils.data import Dataset

from data.rand import Uniform
from data.transforms import Rot90, Flip, Identity, Compose
from data.transforms import GaussianBlur, Noise, Normalize, RandSelect
from data.transforms import RandCrop, CenterCrop, Pad, RandCrop3D, RandomRotion, RandomFlip, RandomIntensityChange
from data.transforms import NumpyType
from data.data_utils import pkload

import numpy as np
import nibabel as nib
import glob

join = os.path.join

HGG = []
LGG = []
for i in range(0, 260):
    HGG.append(str(i).zfill(3))
for i in range(336, 370):
    HGG.append(str(i).zfill(3))
for i in range(260, 336):
    LGG.append(str(i).zfill(3))

# mask_array = np.array([[True, False, False, False], [False, True, False, False], [False, False, True, False], [False, False, False, True],
#                       [True, True, False, False], [True, False, True, False], [True, False, False, True], [False, True, True, False], [False, True, False, True], [False, False, True, True], [True, True, True, False], [True, True, False, True], [True, False, True, True], [False, True, True, True],
#                       [True, True, True, True]])
"""训练集8：2划分"""


# class Brats_loadall_nii(Dataset):
#     def __init__(self, transforms='', root=None, num_cls=4):
#
#         '''Lizy'''
#         patients_dir = glob.glob(join(root, 'vol', '*_vol.npy'))
#         patients_dir.sort(key=lambda x: x.split('/')[-1][:-8])
#         print('###############', len(patients_dir))
#         n_patients = len(patients_dir)
#         pid_idx = np.arange(n_patients)
#         np.random.seed(0)
#         np.random.shuffle(pid_idx)
#
#         volpaths = []  # 训练数据集
#         for i in enumerate(pid_idx):
#             volpaths.append(patients_dir[i])
#         datalist = [x.split('/')[-1].split('_vol')[0] for x in volpaths]  # 训练数据id
#         '''Lizy'''
#
#         self.volpaths = volpaths
#         self.transforms = eval(transforms or 'Identity()')
#         self.names = datalist
#         self.num_cls = num_cls
#         self.modal_ind = np.array([0, 1, 2, 3])
#
#     def __getitem__(self, index):
#
#         volpath = self.volpaths[index]
#         name = self.names[index]
#
#         x = np.load(volpath)
#         segpath = volpath.replace('vol', 'seg')
#         y = np.load(segpath)
#         x, y = x[None, ...], y[None, ...]
#
#         x, y = self.transforms([x, y])  # x:[1, 128, 128, 128, 4], y:[1, 128, 128, 128]
#
#         x = np.ascontiguousarray(
#             x.transpose(0, 4, 1, 2, 3))  # [Bsize,channels,Height,Width,Depth] [1, 4, 128, 128, 128]
#         _, H, W, Z = np.shape(y)  # _, H, W, Z:1, 128, 128, 128
#         y = np.reshape(y, (-1))  # y.shape:(2097152,)
#         one_hot_targets = np.eye(self.num_cls)[y]  # one_hot_targets:[2097152, 4]
#         yo = np.reshape(one_hot_targets, (1, H, W, Z, -1))  # yo:[1, 128, 128, 128, 4]
#         yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))  # yo:[1, 4, 128, 128, 128]  标签和输入的x的输出shape一致
#
#         x = x[:, self.modal_ind, :, :, :]
#
#         x = torch.squeeze(torch.from_numpy(x), dim=0)  # [4, 128, 128, 128]
#         yo = torch.squeeze(torch.from_numpy(yo), dim=0)  # [4, 128, 128, 128]
#
#         return x, yo, name
#
#     def __len__(self):
#         return len(self.volpaths)

class Brats_loadall_nii(Dataset):
    def __init__(self, transforms='', root=None, num_cls=4):

        '''Lizy'''
        patients_dir = glob.glob(os.path.join(root, 'vol', '*_vol.npy'))
        patients_dir.sort(key=lambda x: x.split('/')[-1][:-8])
        print('###############', len(patients_dir))
        n_patients = len(patients_dir)
        pid_idx = np.arange(n_patients)
        np.random.seed(0)
        np.random.shuffle(pid_idx)

        volpaths = []  # 训练数据集
        for i in pid_idx:
            volpaths.append(patients_dir[i])
        datalist = [x.split('/')[-1].split('_vol')[0] for x in volpaths]  # 训练数据id
        '''Lizy'''

        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        self.num_cls = num_cls
        self.modal_ind = np.array([0, 1, 2, 3])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]

        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath)
        x, y = x[None, ...], y[None, ...]

        x, y = self.transforms([x, y])  # x:[1, 128, 128, 128, 4], y:[1, 128, 128, 128]

        x = np.ascontiguousarray(
            x.transpose(0, 4, 1, 2, 3))  # [Bsize,channels,Height,Width,Depth] [1, 4, 128, 128, 128]
        _, H, W, Z = np.shape(y)  # _, H, W, Z:1, 128, 128, 128
        y = np.reshape(y, (-1))  # y.shape:(2097152,)
        one_hot_targets = np.eye(self.num_cls)[y]  # one_hot_targets:[2097152, 4]
        yo = np.reshape(one_hot_targets, (1, H, W, Z, -1))  # yo:[1, 128, 128, 128, 4]
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))  # yo:[1, 4, 128, 128, 128]  标签和输入的x的输出shape一致

        x = x[:, self.modal_ind, :, :, :]

        x = torch.squeeze(torch.from_numpy(x), dim=0)  # [4, 128, 128, 128]
        yo = torch.squeeze(torch.from_numpy(yo), dim=0)  # [4, 128, 128, 128]

        return x, yo, name

    def __len__(self):
        return len(self.volpaths)

"""训练集8：2划分"""


class Brats_loadall_test_nii(Dataset):
    def __init__(self, transforms='', root=None, modal='all', test_file='test.txt'):
        # data_file_path = os.path.join(root, test_file)
        # with open(data_file_path, 'r') as f:
        #     datalist = [i.strip() for i in f.readlines()]
        # datalist.sort()
        # volpaths = []
        # for dataname in datalist:
        #     volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))

        '''Yao'''
        patients_dir = glob.glob(join(root, 'vol', '*_vol.npy'))
        patients_dir.sort(key=lambda x: x.split('/')[-1][:-8])
        n_patients = len(patients_dir)
        pid_idx = np.arange(n_patients)
        np.random.seed(0)
        np.random.shuffle(pid_idx)
        n_fold_list = np.split(pid_idx, 5)

        volpaths = []  # 测试数据集
        for i, fold in enumerate(n_fold_list):
            if i == 0:
                for idx in fold:
                    volpaths.append(patients_dir[idx])
        datalist = [x.split('/')[-1].split('_vol')[0] for x in volpaths]  # 测试数据id
        '''Yao'''

        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0, 1, 2, 3])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath).astype(np.uint8)
        x, y = x[None, ...], y[None, ...]
        x, y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))  # [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)

        x = x[:, self.modal_ind, :, :, :]
        x = torch.squeeze(torch.from_numpy(x), dim=0)
        y = torch.squeeze(torch.from_numpy(y), dim=0)

        return x, y, name

    def __len__(self):
        return len(self.volpaths)


class Brats_loadall_val_nii(Dataset):
    def __init__(self, transforms='', root=None, settype='train', modal='all'):
        data_file_path = os.path.join(root, 'val.txt')
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]
        datalist.sort()
        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname + '_vol.npy'))
        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0, 1, 2, 3])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath).astype(np.uint8)
        x, y = x[None, ...], y[None, ...]
        x, y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))  # [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)
        x = x[:, self.modal_ind, :, :, :]

        x = torch.squeeze(torch.from_numpy(x), dim=0)
        y = torch.squeeze(torch.from_numpy(y), dim=0)

        return x, y, name

    def __len__(self):
        return len(self.volpaths)

