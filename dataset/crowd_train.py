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
    return i, j, crop_h, crop_w, res_h ,res_w


def  cal_innner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right-inner_left, 0.0) * np.maximum(inner_down-inner_up, 0.0)
    return inner_area



class Crowd_train(data.Dataset):
    def __init__(self, root_path, gd_path, crop_size,
                 downsample_ratio, is_gray=False,
                 method='train'):
        self.gd_path = gd_path
        self.root_path = root_path
        self.gd_list = sorted(glob(os.path.join(os.path.join(self.gd_path, 'labels'), '*.npy')))
        self.tir_list = sorted(glob(os.path.join(os.path.join(self.root_path, 'Infrared'), '*.jpg')))
        self.rgb_list = sorted(glob(os.path.join(os.path.join(self.root_path, 'RGB'), '*.jpg')))
        if method not in ['train', 'val']:
            raise Exception("not implement")
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
        gd_path = self.gd_list[item]
        img_rgb = Image.open(rgb_path).convert('RGB')
        img_tir = Image.open(tir_path).convert('RGB')
        if self.method == 'train':
            keypoints = np.load(gd_path)
            return self.train_transform(img_rgb,img_tir, keypoints)
        elif self.method == 'val':
            keypoints = np.load(gd_path)
            img_rgb = self.trans(img_rgb)  # 对图像进行处理
            img_tir = self.trans(img_tir)
            name = os.path.basename(rgb_path).split('.')[0]  # 获得图像名字
            return img_rgb, img_tir, len(keypoints), name

    def train_transform(self, img_rgb, img_tir, keypoints):
        """random crop image patch and find people in it"""
        wd, ht = img_rgb.size
        st_size = min(wd, ht)
        assert st_size >= self.c_size
        assert len(keypoints) > 0
        i, j, h, w, res_h, res_w = random_crop(ht, wd, self.c_size, self.c_size)
        img_rgb = F.crop(img_rgb, i, j, h, w)  # 给定左上角坐标i，j和裁剪长度进行裁剪图像
        img_tir = F.crop(img_tir, i, j, h, w)
        nearest_dis = np.clip(keypoints[:, 2], 4.0, 16.0)  # 将kp内的第三个数值（距离）裁剪到4-128范围内
        # 人头中心点为中心的边界框
        points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
        points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
        bbox = np.concatenate((points_left_up, points_right_down), axis=1)
        # 获取bbox区域面积
        inner_area = cal_innner_area(j, i, j+w, i+h, bbox)  # j为y方向，i为x方向
        origin_area = nearest_dis * nearest_dis
        ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
        mask = (ratio >= 0.3)
        # 过滤掉处于裁剪边界的点
        target = ratio[mask]   # 注释点处于图像中心的概率
        keypoints = keypoints[mask]
        keypoints = keypoints[:, :2] - [j, i]  # change coodinate 使左上角坐标为0
        if len(keypoints) > 0:
            if random.random() > 0.5:  # 随机翻转
                img_rgb = F.hflip(img_rgb)
                img_tir = F.hflip(img_tir)
                keypoints[:, 0] = w - keypoints[:, 0]   # 图像翻转、注释翻转
            """
            if random.random() > 0.5:  # 随机翻转
                img_rgb = F.vflip(img_rgb)
                img_tir = F.vflip(img_tir)
                keypoints[:, 1] = h - keypoints[:, 1]   # 图像翻转、注释翻转
            
            if random.random() > 0.5:  # 随机翻转
                img_rgb = F.resize(img_rgb, [384, 384])
                img_tir = F.resize(img_tir, [384, 384])
                keypoints[:, :2] = keypoints[:, :2] * 0.75   # 图像翻转、注释翻转
                """
        """
        else:
            if random.random() > 0.5:
                img_rgb = F.hflip(img_rgb)
                img_tir = F.hflip(img_tir)   # 无注释
        """
        return self.trans(img_rgb), self.trans(img_tir), torch.from_numpy(keypoints.copy()).float(), \
               torch.from_numpy(target.copy()).float(), st_size
