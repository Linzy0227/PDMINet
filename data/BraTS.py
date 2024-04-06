import os
import torch
from torch.utils.data import Dataset
import random
# random.seed(1000)
import numpy as np
# np.random.seed(1000)
from torchvision.transforms import transforms
import pickle
from scipy import ndimage
import torch.nn.functional as F
import glob


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


class MaxMinNormalization(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        Max = np.max(image)
        Min = np.min(image)
        image = (image - Min) / (Max - Min)

        return {'image': image, 'label': label}


class Random_Flip(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = np.flip(image, 0)
            label = np.flip(label, 0)
        if random.random() < 0.5:
            image = np.flip(image, 1)
            label = np.flip(label, 1)
        if random.random() < 0.5:
            image = np.flip(image, 2)
            label = np.flip(label, 2)

        return {'image': image, 'label': label}


class Random_Crop(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        H = random.randint(0, 240 - 128)
        W = random.randint(0, 240 - 128)
        D = random.randint(0, 160 - 128)

        image = image[H: H + 128, W: W + 128, D: D + 128, ...]
        label = label[..., H: H + 128, W: W + 128, D: D + 128]  # [128, 128, 128]

        return {'image': image, 'label': label}


class Random_intencity_shift(object):
    def __call__(self, sample, factor=0.1):
        image = sample['image']
        label = sample['label']

        scale_factor = np.random.uniform(1.0-factor, 1.0+factor, size=[1, image.shape[1], 1, image.shape[-1]])
        shift_factor = np.random.uniform(-factor, factor, size=[1, image.shape[1], 1, image.shape[-1]])

        image = image*scale_factor+shift_factor

        return {'image': image, 'label': label}


class Random_rotate(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        angle = round(np.random.uniform(-10, 10), 2)
        image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False)
        label = ndimage.rotate(label, angle, axes=(0, 1), reshape=False)

        return {'image': image, 'label': label}


class Pad(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')  # (H,W,D,4)
        label = np.pad(label, ((0, 0), (0, 0), (0, 5)), mode='constant')
        return {'image': image, 'label': label}
    #(240,240,155)>(240,240,160)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))  # image: (H,W,D,C) -> (C,H,W,D)
        label = sample['label']
        label = np.ascontiguousarray(label)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return {'image': image, 'label': label}


def transform(sample):  # 训练集
    trans = transforms.Compose([
        Pad(),
        # Random_rotate(),  # time-consuming
        Random_Crop(),  # 裁剪
        Random_Flip(),  # 翻转
        Random_intencity_shift(),  # 随机密度改变
        ToTensor()  # 数据类型转换
    ])

    return trans(sample)


def transform_valid(sample):  # 验证集
    trans = transforms.Compose([
        Pad(),
        # MaxMinNormalization(),
        ToTensor()
    ])

    return trans(sample)


class BraTS(Dataset):
    """
    data_path: 预处理后的数据集所在地址
    file_path: 对应的.txt文件所在地址
    """
    def __init__(self, file_path, data_path='', mode=''):  # file_path:path of xxx.txt, data_path:path of data
        self.lines = []
        paths, names = [], []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                name = line
                names.append(name)
                path = os.path.join(data_path, line, name + '_')
                paths.append(path)
                self.lines.append(line)
        self.mode = mode
        self.paths = paths
        self.names = names
        self.modal_ind = np.array([0, 1, 2, 3])

    def __getitem__(self, item):
        path = self.paths[item]
        if self.mode == 'train':
            image, label = pkload(path + 'data_f32b0.pkl')
            sample = {'image': image, 'label': label}
            sample['label'][sample['label'] == 4] = 3
            sample = transform(sample)
            img = sample['image']
            gt = sample['label']
            H, W, Z = gt.size()
            gt_onehot = F.one_hot(gt, num_classes=4)
            gt_onehot = gt_onehot.view(H, W, Z, -1).permute(3, 0, 1, 2).contiguous()
            return img, gt_onehot

        elif self.mode == 'valid':
            # 验证集
            image, label = pkload(path + 'data_f32b0.pkl')
            sample = {'image': image, 'label': label}
            sample = transform_valid(sample)
            return sample['image'], sample['label']
        else:
            image, label = pkload(path + 'data_f32b0.pkl')
            # image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
            image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
            image = torch.from_numpy(image).float()
            return image

    def __len__(self):
        return len(self.paths)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]



if __name__=='__main__':
    # train_list = os.path.join('/Users/lzzzy/Desktop/MyExperiments/DataSet/BraTS2021', 'train_valid', 'train_valid.txt')
    # train_root = os.path.join('/Users/lzzzy/Desktop/MyExperiments/DataSet/BraTS2021', 'train_valid')
    # train_set = BraTS(train_root, 'train')
    path = '/Users/lzzzy/Desktop/TestCode/'
    image, label = pkload(path + 'Brats18_2013_2_1_data_f32b0.pkl')
    # image, label = pkload('/Users/lzzzy/Desktop/MyExperiments/DataSet/BraTS2021/train_valid/BraTS2021_00000/BraTS2021_00000_' + 'data_f32b0.pkl')
    image.shape
    label.shape
    sample = {'image': image,'label': label}
    trans = transforms.Compose([Random_Crop()
                                 ])
    out=trans(sample)
    print(out['image'].shape)
    print(image.shape)

