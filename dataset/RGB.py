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


class RGB(data.Dataset):
    def __init__(self, data_path, mode, main_transform=None, img_transform=None, rgb_img_transform=None,
                 gt_transform=None):
        self.img_path = data_path + '/Visible'
        self.gt_path = data_path + '/maps_fixed_kernel'
        self.data_files = [filename for filename in os.listdir(self.img_path) \
                           if os.path.isfile(os.path.join(self.img_path, filename))]
        self.num_samples = len(self.data_files)
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.rgb_img_transform = rgb_img_transform

        self.mode = mode

    def __getitem__(self, index):
        fname = self.data_files[index]
        img, den = self.read_image_and_gt(fname)
        if self.main_transform is not None:
            img, den = self.main_transform(img, den)

        if self.rgb_img_transform is not None:
            img = self.img_transform(img)
        if self.gt_transform is not None:
            den = self.gt_transform(den)

        # if self.mode == 'train':
        #     img, den = self.random_crop(img, den)
        # else:
        #     img, den = self.crop(img, den, (424, 760))

        return img, den

    def __len__(self):
        return self.num_samples

    def crop(self, img, den, dst_size):
        return img[:, :dst_size[0], :dst_size[1]], den[:dst_size[0], :dst_size[1]]

    def random_crop(self, img, den, dst_size=(296, 536)):
        # dst_size: ht, wd

        _, ts_hd, ts_wd = img.shape

        # print(ts_wd - dst_size[1], ts_hd - dst_size[0])

        x1 = random.randint(0, ts_wd - dst_size[1])
        y1 = random.randint(0, ts_hd - dst_size[0])
        x2 = x1 + dst_size[1]
        y2 = y1 + dst_size[0]

        return img[:, y1:y2, x1:x2], den[y1:y2, x1:x2]

    def read_image_and_gt(self, fname):
        img = Image.open(os.path.join(self.img_path, fname))
        if img.mode == 'L':
            img = img.convert('RGB')

        den = h5py.File(os.path.join(self.gt_path, fname.replace('.jpg', '.h5')), 'r')
        den = np.asarray(den['density'])
        # den = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).values

        den = den.astype(np.float32)
        den = Image.fromarray(den)
        return img, den

    def get_num_samples(self):
        return self.num_samples
