import numpy as np
import os
import random
from scipy import io as sio
import sys
import torch
from torch.utils import data
from PIL import Image, ImageOps
import h5py

import pandas as pd

from config import cfg

additional_path = '/home/mawenya/liqing/VDCC/Train'


class RGBIR(data.Dataset):
    def __init__(self, data_path, mode, main_transform=None, img_transform=None, rgb_img_transform=None,
                 gt_transform=None):
        self.img_path = data_path + '/Infrared'
        self.add_img_path = additional_path + '/Infrared'
        # self.img_path = data_path + '/fake'
        # self.rgb_path = data_path + '/Visible'
        # self.gt_path = data_path + '/maps_fixed_kernel'
        self.data_files = [os.path.join(self.img_path, filename) for filename in os.listdir(self.img_path) \
                           if os.path.isfile(os.path.join(self.img_path, filename))]
        if mode == 'train':
            self.data_files = self.data_files + [os.path.join(self.add_img_path, filename) for filename in
                                                 os.listdir(self.add_img_path) \
                                                 if os.path.isfile(os.path.join(self.add_img_path, filename))]
        self.num_samples = len(self.data_files)
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.rgb_img_transform = rgb_img_transform

        self.mode = mode

    def __getitem__(self, index):
        fname = self.data_files[index]
        rgb_img, img, den = self.read_image_and_gt(fname)
        if self.main_transform is not None:
            imgs, den = self.main_transform([rgb_img, img], den)
            rgb_img, img = imgs
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.rgb_img_transform is not None:
            rgb_img = self.img_transform(rgb_img)
        if self.gt_transform is not None:
            den = self.gt_transform(den)

        if self.mode == 'train':
            rgb_img, img, den = self.random_crop(rgb_img, img, den)

        return rgb_img, img, den

    def __len__(self):
        return self.num_samples

    def random_crop(self, img, infrared, den, dst_size=(512, 640)):
        # dst_size: ht, wd

        _, ts_hd, ts_wd = img.shape

        x1 = random.randint(0, ts_wd - dst_size[1])
        y1 = random.randint(0, ts_hd - dst_size[0])
        x2 = x1 + dst_size[1]
        y2 = y1 + dst_size[0]

        return img[:, y1:y2, x1:x2], infrared[:, y1:y2, x1:x2], den[y1:y2, x1:x2]

    def read_image_and_gt(self, fname):
        img = Image.open(os.path.join(self.img_path, fname))
        if img.mode == 'L':
            img = img.convert('RGB')

        if 'fake' in fname:
            den = h5py.File(os.path.join(fname.replace('/Infrared', '/maps_fixed_kernel').replace('_fake.png', '.h5')),
                            'r')
            rbg_img = Image.open(fname.replace('/Infrared', '/Visible').replace('_fake.png', '.jpg'))
        else:

            rbg_img = Image.open(fname.replace('/Infrared', '/Visible').replace('R.jpg', '.jpg'))
            den = h5py.File(fname.replace('/Infrared', '/maps_fixed_kernel').replace('R.jpg', '.h5'), 'r')

        # den = h5py.File(os.path.join(self.gt_path, fname.replace('_fake.png', '.h5')), 'r')
        den = np.asarray(den['density'])
        # den = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).values

        den = den.astype(np.float32)
        den = Image.fromarray(den)
        return rbg_img, img, den

    def get_num_samples(self):
        return self.num_samples
