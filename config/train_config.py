from easydict import EasyDict as edict
import time

# init
__C = edict()
cfg = __C

#------------------------------TRAIN------------------------
__C.SEED = 3035 # random seed,  for reproduction
__C.DATA_DIR = r'/data/che_xiao/crowd_count/外部数据集/DroneRGBT/DroneRGBT1/Train'
__C.GT_DIR = r'/data/che_xiao/crowd_count/外部数据集/DroneRGBT/DroneRGBT1/Train'
__C.SAVE_DIR = r'/data/che_xiao/crowd_count/counter/checkpoints'
__C.VAL_DATA_DIR = r'/data/che_xiao/crowd_count/外部数据集/DroneRGBT/DroneRGBT1/Test'
__C.VAL_GT_DIR = r'/data/che_xiao/crowd_count/外部数据集/DroneRGBT/DroneRGBT1/Test'


__C.RESUME = False # contine training
__C.RESUME_PATH = '/data/che_xiao/crowd_count/counter/checkpoints/pth/6.6.pth'

__C.LR = 5e-4 # learning rate 1e-5
__C.WEIGHT_DECAY = 0.05
__C.LR_DECAY = 0.995 # decay rate
__C.LR_DECAY_START = -1 # when training epoch is more than it, the learning rate will be begin to decay
__C.NUM_EPOCH_LR_DECAY = 1 # decay frequency
__C.MAX_EPOCH = 1000
__C.VAL_START = 300
__C.VAL_EPOCH = 30
__C.BATCH_SIZE = 12
__C.NUM_WORKERS = 2
__C.DOWNSAMPLE_RATIO = 1
__C.CROP_SIZE = 512
__C.IS_GRAY = False


# use background
__C.USE_BACKGROUND = True
__C.SIGMA = 5
__C.BACKGROUND_RATIO = 1

__C.GPU_ID = [0, 1, 2, 3] # sigle gpu: [0], [1] ...; multi gpus: [0,1]

# multi-task learning weights, no use for single model, such as MCNN, VGG, VGG_DECODER, Res50, CSRNet, and so on

__C.LAMBDA_1 = 1e-4# SANet:0.001 CMTL 0.0001

# print
__C.PRINT_FREQ = 10

now = time.strftime("%m-%d_%H-%M", time.localtime())


#------------------------------VAL------------------------
__C.VAL_DENSE_START = -1
__C.VAL_FREQ = 10 # Before __C.VAL_DENSE_START epoches, the freq is set as __C.VAL_FREQ

#------------------------------VIS------------------------
__C.VISIBLE_NUM_IMGS = 1 #  must be 1 for training images with the different sizes



#================================================================================
#================================================================================
#================================================================================
