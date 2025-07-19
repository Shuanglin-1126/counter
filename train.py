from utils.trainer_m import RegTrainer_m
from config.train_config import cfg
import os
import torch
args = None


torch.backends.cudnn.benchmark = True    # 使用torch的加速策略，根据硬件选择最快算法
gpu_ids = ','.join(map(str, cfg.GPU_ID))
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
#os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU_ID.strip()  # set vis gpu
trainer = RegTrainer_m()   # from utils.regression_trainer import RegTrainer
trainer.setup()
trainer.train()    # RegTrainer里的函数

