import torch
import os
import numpy as np
from dataset.crowd_test import Crowd_test
from  models.MMCNN_FPN import CrowdCounter_fpn
from config.test_config import cfg
from utils.logger import setlogger
import logging





if __name__ == '__main__':
    setlogger(os.path.join(cfg.SAVE_DIR, 'test.log'))  # set logger
    gpu_ids = ','.join(map(str, cfg.GPU_ID))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    datasets = Crowd_test(cfg.DATA_DIR,
                        cfg.CROP_SIZE,
                        cfg.DOWNSAMPLE_RATIO,
                        cfg.IS_GRAY, 'test')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=8, pin_memory=False)
    model = CrowdCounter_fpn(cfg.GPU_ID)
    device = torch.device('cuda')
    model.to(device)
    #model.load_state_dict(torch.load(os.path.join(cfg.SAVE_DIR, 'best_model.pth'), device))
    model.load_state_dict(torch.load(r'/data/che_xiao/crowd_count/counter/checkpoints/pth/6.8rpn.pth', device))
    epoch_counts = []

    for imgs_rgb, imgs_tir, name in dataloader:
        imgs_rgb = imgs_rgb.to(device)
        imgs_tir = imgs_tir.to(device)
        assert imgs_tir.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs = model.test_forward([imgs_rgb, imgs_tir])
            epoch_count = torch.sum(outputs).item()
            epoch_counts.append(epoch_count)
            logging.info('name {}   human count: {:.2f}'
                         .format(name, epoch_count))

