import torch
import torch.nn as nn
from torchvision import models

import torch.nn.functional as F
import pdb

from collections import OrderedDict

from misc.layer import Conv2d, FC
from misc.utils import *

model_path = r'/data/che_xiao/crowd_count/counter/resnet50-19c8e357.pth'

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        #print("Channel attention shape : {}".format(out.size()))
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        #print("spatial attention shape : {}".format(x.size()))
        return self.sigmoid(x)

class ModalityAttention(nn.Module):
    def __init__(self, in_planes):
       super(ModalityAttention, self).__init__()
       self.in_planes = in_planes
       self.mod_t = SpatialAttention()
       self.mod_r = SpatialAttention()
       self.cross = ChannelAttention(in_planes*2)

    def forward(self, r, t):
        atten_r = self.mod_r(r) * r
        atten_t = self.mod_t(t) * t

        m = torch.cat([r, t], 1)
        atten_m = torch.cat([atten_r, atten_t], 1)

        feat = self.cross(m) * atten_m

        return feat[:, :self.in_planes, :, :], feat[:, self.in_planes:, :, :]


class Res50(nn.Module):
    def __init__(self, pretrained=True):
        super(Res50, self).__init__()

        # ****************RGB_para****************
        self.RGB_para1_1x1 = Conv2d(64, 256, 1, same_padding=True, NL='relu')
        self.RGB_para2_1x1 = nn.Sequential(Conv2d(256, 512, 1, same_padding=True, NL='relu'), nn.MaxPool2d(kernel_size=2, stride=2))

        # *********T_para**********************
        self.T_para1_1x1 = Conv2d(64, 256, 1, same_padding=True, NL='relu')
        self.T_para2_1x1 = nn.Sequential(Conv2d(256, 512, 1, same_padding=True, NL='relu'), nn.MaxPool2d(kernel_size=2, stride=2))


        # *** prediction****
        self.de_pred1 = Conv2d(1024, 128, 1, same_padding=True, NL='relu')
        self.de_pred2 = Conv2d(128, 1, 1, same_padding=True, NL='relu')

        self.reduce = Conv2d(1024, 512, 1, same_padding=True, NL='relu')

        self.rpnconv1 = Conv2d(128, 1, 1, same_padding=True, NL='relu')
        self.rpnconv2 = Conv2d(512, 1, 1, same_padding=True, NL='relu')
        self.rpnconv3 = Conv2d(1024, 1, 1, same_padding=True, NL='relu')
        self.fin = Conv2d(4, 1, 1, same_padding=True, NL='relu')


        self.ma1 = ModalityAttention(64)
        self.ma2 = ModalityAttention(256)
        self.ma3 = ModalityAttention(512)


        initialize_weights(self.modules())

        # ***basic resnet****
        pre_wts = torch.load(model_path)

        self.conv1_rgb = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv1_t = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )


        self.conv2_x = make_res_layer(Bottleneck, 64, 64, 3, 1)
        self.conv3_x = make_res_layer(Bottleneck, 256, 128, 4, 2)
        self.conv4_x = make_res_layer(Bottleneck, 512, 256, 6, 1)


        state_conv_dict = OrderedDict()
        state_layer1_dict = OrderedDict()
        state_layer2_dict = OrderedDict()
        state_layer3_dict = OrderedDict()
        for k, v in pre_wts.items():
            if 'layer1' in k:
                #print(k)
                name = k[7:]
                state_layer1_dict[name] = v
                if 'bn' in name and 'downsample' not in name:
                    idx = name.find('bn')
                    x = name[idx:idx+3]
                    state_layer1_dict[name.replace(x, x+'_target')] = v
                if 'downsample' in name:
                    state_layer1_dict[name.replace('downsample', 'downsample_target')] = v

            elif 'layer2' in k:
                name = k[7:]
                state_layer2_dict[name] = v
                if 'bn' in name and 'downsample' not in name:
                    idx = name.find('bn')
                    x = name[idx:idx + 3]
                    state_layer2_dict[name.replace(x, x+'_target')] = v
                if 'downsample' in name:
                    state_layer2_dict[name.replace('downsample', 'downsample_target')] = v

            elif 'layer3' in k:
                name = k[7:]
                state_layer3_dict[name] = v
                if 'bn' in name and 'downsample' not in name:
                    idx = name.find('bn')
                    x = name[idx:idx + 3]
                    state_layer3_dict[name.replace(x, x+'_target')] = v
                if 'downsample' in name:
                    state_layer3_dict[name.replace('downsample', 'downsample_target')] = v

            elif 'layer' not in k and 'fc' not in k:
                if 'conv' in k:
                    k = k.replace('conv1', '0')
                    #print(k)
                if 'bn1' in k:
                    k = k.replace('bn1', '1')
                state_conv_dict[k] = v

        #print(state_layer1_dict.keys(), self.conv2_x.state_dict().keys())

        self.conv1_rgb.load_state_dict(state_conv_dict)
        self.conv1_t.load_state_dict(state_conv_dict)
        self.conv2_x.load_state_dict(state_layer1_dict)
        self.conv3_x.load_state_dict(state_layer2_dict)
        self.conv4_x.load_state_dict(state_layer3_dict)

    def forward(self, img):

        rgb, ir = img
        featR = self.conv1_rgb(rgb)
        featT = self.conv1_t(ir)

        featR, featT = self.ma1(featR, featT)


        feat_p1 = torch.cat([featT, featR], 1)
        feat_p1 = self.rpnconv1(feat_p1)
        feat_p1 = F.upsample(feat_p1, scale_factor=4)

        # **********block 1*********
        feat_T = self.T_para1_1x1(featT)
        feat_R = self.RGB_para1_1x1(featR)

        feat_MT = self.conv2_x(featT)
        for layer in self.conv2_x:
            featR = layer(featR, target=True)
        feat_MR = featR

        featT = feat_MT + feat_T
        featR = feat_MR + feat_R
        featR, featT = self.ma2(featR, featT)

        feat_p2 = torch.cat([featT, featR], 1)
        feat_p2 = self.rpnconv2(feat_p2)
        feat_p2 = F.upsample(feat_p2, scale_factor=4)

        # *********block 2 ********
        feat_T = self.T_para2_1x1(featT)
        feat_R = self.RGB_para2_1x1(featR)

        feat_MT = self.conv3_x(featT)
        for layer in self.conv3_x:
            #print(layer)
            featR = layer(featR, target=True)
        feat_MR = featR

        featR = feat_MR + feat_R
        featT = feat_MT + feat_T
        featR, featT = self.ma3(featR, featT)

        feat_p3 = torch.cat([featT, featR], 1)
        feat_p3 = self.rpnconv3(feat_p3)
        feat_p3 = F.upsample(feat_p3, scale_factor=8)

        # ***********fusion************
        feat = torch.cat([featT, featR], 1)
        feat = self.reduce(feat)



        # ********block3 **********

        conv4_feat = self.conv4_x(feat)
        de_pred1_feat = self.de_pred1(conv4_feat)
        de_pred2_feat = self.de_pred2(de_pred1_feat)
        feat_p4 = F.upsample(de_pred2_feat, scale_factor=8)

        feat = torch.cat([feat_p1, feat_p2, feat_p3, feat_p4], 1)
        feat = self.fin(feat)

        return feat



def make_res_layer(block, inplanes, planes, blocks, stride=1):

    downsample = None
    downsample_target = None
    #inplanes=512
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )
        downsample_target = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample, downsample_target))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, downsample_target=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.downsample_target = downsample_target
        self.stride = stride

        self.bn1_target = nn.BatchNorm2d(planes)
        self.bn2_target = nn.BatchNorm2d(planes)
        self.bn3_target = nn.BatchNorm2d(planes * self.expansion)

    def forward(self, x, target=False):
        residual = x

        out = self.conv1(x)
        if not target:
            out = self.bn1(out)
        else:
            out = self.bn1_target(out)
        out = self.relu(out)

        out = self.conv2(out)
        if not target:
            out = self.bn2(out)
        else:
            out = self.bn2_target(out)
        out = self.relu(out)

        out = self.conv3(out)
        if not target:
            out = self.bn3(out)
        else:
            out = self.bn3_target(out)

        if self.downsample is not None:
            if not target:
                residual = self.downsample(x)
            else:
                residual = self.downsample_target(x)

        out += residual
        out = self.relu(out)

        return out


class CrowdCounter_fpn(nn.Module):
    def __init__(self, gpus):
        super(CrowdCounter_fpn, self).__init__()

        self.CCN = Res50()
        if len(gpus) > 1:
             self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
             self.CCN = self.CCN.cuda()
        self.loss_mse_fn = nn.MSELoss().cuda()

    def fix_bn(self, m):
        classname = m.__class__.__name__
        # print(classname)
        if classname.find('BatchNorm') != -1:
            m.eval()

    @property
    def loss(self):
        return self.loss_mse

    def supervise_forward(self, img, real_img):  #during training GANs
        density_map = self.CCN(img)
        density_map_gt = self.CCN(real_img.detach())
        self.loss_mse = self.build_loss(density_map.squeeze(), density_map_gt.squeeze())
        return density_map

    def train_forward(self, img):   # during training Counter
        density_map = self.CCN(img)
        return density_map


    def build_loss(self, density_map, gt_data):
        loss_mse = self.loss_mse_fn(density_map, gt_data)
        return loss_mse

    def test_forward(self, img):
        density_map = self.CCN(img)
        return density_map

