from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)  # 生成0-res内的随机整数
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


def  cal_innner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right-inner_left, 0.0) * np.maximum(inner_down-inner_up, 0.0)
    return inner_area



class Crowd_test(data.Dataset):
    def __init__(self, root_path, crop_size,
                 downsample_ratio, is_gray=False,
                 method='test'):
        self.root_path = root_path
        self.tir_list = sorted(glob(os.path.join(os.path.join(self.root_path,'tir'),'*.jpg')))
        self.rgb_list = sorted(glob(os.path.join(os.path.join(self.root_path,'rgb'),'*.jpg')))
        self.method = method
        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0   # crop_size能被d_ratio整除
        self.dc_size = self.c_size // self.d_ratio  # 下采样后尺寸

        if is_gray:
            self.trans = transforms.Compose([
                transforms.ToTensor(),  # 将图像或数组转为张量
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化
            ])  # 将命令按顺序应用至数组
        else:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.rgb_list)

    def __getitem__(self, item):
        rgb_path = self.rgb_list[item]
        tir_path = self.tir_list[item]
        img_rgb = Image.open(rgb_path).convert('RGB')
        img_tir = Image.open(tir_path).convert('RGB')
        img_rgb = self.trans(img_rgb)  # 对图像进行处理
        img_tir = self.trans(img_tir)
        name = os.path.basename(rgb_path).split('.')[0]  # 获得图像名字
        return img_rgb, img_tir, name


