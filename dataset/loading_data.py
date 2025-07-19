import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import misc.transforms as own_transforms
from .RGBIR import RGBIR
from .RGB import RGB
from .setting import cfg_data
import torch
import random


def loading_data():
    mean_std = cfg_data.MEAN_STD
    rgb_mean_std = cfg_data.RGB_MEAN_STD
    log_para = cfg_data.LOG_PARA
    factor = cfg_data.LABEL_FACTOR
    train_main_transform = own_transforms.Compose([
        own_transforms.RandomHorizontallyFlip()
    ])
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    rgb_img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*rgb_mean_std)
    ])
    gt_transform = standard_transforms.Compose([
        own_transforms.GTScaleDown(factor),
        own_transforms.LabelNormalize(log_para)
    ])
    restore_transform_rgb = standard_transforms.Compose([
        own_transforms.DeNormalize(*rgb_mean_std),
        standard_transforms.ToPILImage()
    ])
    restore_transform_ir = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

    train_set = RGBIR(cfg_data.DATA_PATH + '/Train', 'train', main_transform=train_main_transform,
                     img_transform=img_transform, rgb_img_transform=rgb_img_transform, gt_transform=gt_transform)
    train_loader = DataLoader(train_set, batch_size=cfg_data.TRAIN_BATCH_SIZE, num_workers=8,
                                  shuffle=True, drop_last=True)

    val_set = RGBIR(cfg_data.DATA_PATH + '/Test', 'test', main_transform=None, img_transform=img_transform, rgb_img_transform=rgb_img_transform,
                   gt_transform=gt_transform)
    val_loader = DataLoader(val_set, batch_size=cfg_data.VAL_BATCH_SIZE, num_workers=8, shuffle=True, drop_last=False)

    extra_train_set = RGB(cfg_data.EXTRA_DATA_PATH + '/Train', 'train', main_transform=train_main_transform,
                     img_transform=img_transform, rgb_img_transform=rgb_img_transform, gt_transform=gt_transform)
    extra_train_loader = DataLoader(extra_train_set, batch_size=cfg_data.TRAIN_BATCH_SIZE, num_workers=8,
                                  shuffle=True, drop_last=True)

    extra_val_set = RGB(cfg_data.EXTRA_DATA_PATH + '/Test', 'test', main_transform=None, img_transform=img_transform, rgb_img_transform=rgb_img_transform,
                   gt_transform=gt_transform)
    extra_val_loader = DataLoader(extra_val_set, batch_size=cfg_data.VAL_BATCH_SIZE, num_workers=8, shuffle=True, drop_last=False)

    return train_loader, val_loader, extra_train_loader, extra_val_loader, restore_transform_rgb, restore_transform_ir
