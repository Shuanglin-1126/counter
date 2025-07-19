from utils.helpers import Save_Handle, AverageMeter
import os
import sys
from utils.logger import setlogger
import time
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import logging
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from  models.MMCNN_FPN import CrowdCounter_fpn
from dataset.crowd_train import Crowd_train
from losses.losses_bay import Bay_Loss
from losses.post_prob import Post_Prob
from config.train_config import cfg
from torch.optim.lr_scheduler import StepLR


def train_collate(batch):
    transposed_batch = list(zip(*batch))   # 将数据打包成元组，并返回元组组成的列表
    images_rgb = torch.stack(transposed_batch[0], 0)
    images_tir = torch.stack(transposed_batch[1], 0)
    points = transposed_batch[2]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[3]
    st_sizes = torch.FloatTensor(transposed_batch[4])
    return images_rgb, images_tir, points, targets, st_sizes

class RegTrainer(object):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        setlogger(os.path.join(cfg.SAVE_DIR, 'train.log'))  # set logger
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            # for code conciseness, we release the single gpu version
            #assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        # from datasets.crowd import Crowd 输出imgs、points、targets、st_size
        self.datasets = {x: Crowd_train(cfg.DATA_DIR if x == 'train' else cfg.VAL_DATA_DIR,
                                      cfg.GT_DIR if x == 'train' else cfg.VAL_GT_DIR,
                                  cfg.CROP_SIZE,
                                  cfg.DOWNSAMPLE_RATIO,
                                  cfg.IS_GRAY, x) for x in ['train','val']}
        # 定义数据处理器
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(cfg.BATCH_SIZE
                                                      if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=cfg.NUM_WORKERS * self.device_count,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val']}
        self.model = CrowdCounter_fpn(cfg.GPU_ID)  # 使用模型
        self.model.to(self.device)  # 将模型加载至指定设备
        # 定义优化器
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

        self.scheduler = StepLR(self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)

        self.start_epoch = 0
        # 加载预训练权重文件
        if cfg.RESUME:
            suf = cfg.RESUME_PATH.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(cfg.RESUME_PATH, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(cfg.RESUME_PATH, self.device))

        self.post_prob = Post_Prob(cfg.SIGMA,
                                   cfg.CROP_SIZE,
                                   cfg.DOWNSAMPLE_RATIO,
                                   cfg.BACKGROUND_RATIO,
                                   cfg.USE_BACKGROUND,
                                   self.device)
        self.criterion = Bay_Loss(cfg.USE_BACKGROUND, self.device)
        self.save_list = Save_Handle(max_num=1)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_count = 0

    def train(self):
        """training process"""
        for epoch in range(self.start_epoch, cfg.MAX_EPOCH):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, cfg.MAX_EPOCH - 1) + '-' * 5)
            if epoch > cfg.LR_DECAY_START:
                self.scheduler.step()
            self.epoch = epoch
            self.train_eopch(epoch)
            # 中间进行验证
            if epoch % cfg.VAL_EPOCH == 0 or epoch >= cfg.VAL_START:
                self.val_epoch()

    def train_eopch(self,epoch):
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode

        # Iterate over data 迭代数据.
        for step, (imgs_rgb, imgs_tir, points, targets, st_sizes) in enumerate(self.dataloaders['train']):
            imgs_rgb = imgs_rgb.to(self.device)
            imgs_tir = imgs_tir.to(self.device)
            st_sizes = st_sizes.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            targets = [t.to(self.device) for t in targets]

            with torch.set_grad_enabled(True):
                pred_map = self.model.train_forward([imgs_rgb, imgs_tir])
                prob_list = self.post_prob(points, st_sizes)
                loss = self.criterion(prob_list, targets, pred_map)
                # torch训练流程，梯度清0，反向传播，根据梯度更新模型参数
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # 计算损失
                N = imgs_rgb.size(0)
                pre_count = torch.sum(pred_map.view(N, -1), dim=1).detach().cpu().numpy()
                res = pre_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)

        logging.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                             time.time() - epoch_start))
        mse = np.sqrt(epoch_mse.get_avg())
        mae = epoch_mae.get_avg()
        if epoch % cfg.VAL_EPOCH == 0:
            model_state_dic = self.model.state_dict()
            save_path = os.path.join(cfg.SAVE_DIR, '{}_ckpt.tar'.format(self.epoch))
            torch.save({
                'epoch': self.epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_state_dict': model_state_dic
            }, save_path)
            self.save_list.append(save_path)  # control the number of saved models
            """
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            model_state_dic = self.model.state_dict()
            logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                    self.best_mae,
                                                                                    self.epoch))
            torch.save(model_state_dic, os.path.join(cfg.SAVE_DIR, 'best_model.pth'))
            """

    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        # Iterate over data.
        for imgs_rgb, imgs_tir, count, name in self.dataloaders['val']:
            imgs_rgb = imgs_rgb.to(self.device)
            imgs_tir = imgs_tir.to(self.device)
            # inputs are images with different sizes
            assert imgs_tir.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs = self.model.train_forward([imgs_rgb, imgs_tir])
                res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, mse, mae, time.time() - epoch_start))

        model_state_dic = self.model.state_dict()
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                 self.best_mae,
                                                                                 self.epoch))
            torch.save(model_state_dic, os.path.join(cfg.SAVE_DIR, 'best_model.pth'))