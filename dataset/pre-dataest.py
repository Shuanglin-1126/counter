import time
from tqdm import trange
import numpy as np
import os
import argparse
import xml.dom.minidom as xmlmm

def find_dis(point):
    """
    获取每个点和距离它最近的三个点之间的平均距离
    :param point: 点注释
    :return: 每个点的平均距离
    """
    square = np.sum(point*point, axis=1)     # 每个点的距离；x方加y方  N x 2
    dis1 = np.sqrt(np.maximum(square[:, None] - 2*np.matmul(point, point.T)
                             + square[None, :], 0.0))    # 不同点之间的距离
    m, n = dis1.shape
    if m >= 4:
        dis = np.partition(dis1, 3, axis=1)[:, 0:4]  # np.partition排序取每行最小的四个放前面
        dis = np.sort(dis)
        dis = np.mean(dis[:, 1:4],axis=1, keepdims=True)
    elif m == 3:
        dis = np.sort(dis1)
        dis = np.mean(dis[:, 1:3],axis=1, keepdims=True)
    elif m == 2:
        dis = np.sort(dis1)
        dis = dis[:, 1]
    elif m == 1:
        dis = np.zeros(1)
        dis[0,] = np.float32(32)
    return dis

def generate_data(data_path):
    dom = xmlmm.parse(data_path)
    root = dom.documentElement
    node_x = root.getElementsByTagName('x')
    node_y = root.getElementsByTagName('y')
    points_x = []
    points_y = []
    for index, point_x in enumerate(node_x):
        value = point_x.childNodes[0].nodeValue
        points_x.append(np.float32(value))
    for index, point_y in enumerate(node_y):
        value = point_y.childNodes[0].nodeValue
        points_y.append(np.float32(value))
    points = np.array([points_x, points_y]).T
    return points

def parse_args():
    parser = argparse.ArgumentParser(description='preprocess-dataests')
    parser.add_argument('--train-dir', default=r'/',
                        help='original data directory')
    parser.add_argument('--test-dir', default=r'',
                        help='processed data directory')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()       # 输入参数
    sub_data_dir = os.path.join(args.train_dir, 'GT_')    # 获取原始图片路径
    sub_save_dir = os.path.join(args.test_dir, 'labels')
    if not os.path.exists(sub_save_dir):
        os.makedirs(sub_save_dir)
    data_paths = [os.path.join(sub_data_dir, file) for file in os.listdir(sub_data_dir)]
    for i in trange(len(data_paths)):
        data_path = data_paths[i]
        name = os.path.basename(data_path)
        points = generate_data(data_path)
        dis = find_dis(points)
        dis = dis.reshape(-1,1)
        points = np.concatenate((points, dis), axis=1)
        gd_save_path = os.path.join(sub_save_dir, name)
        gd_save_path = gd_save_path.replace('R.xml', '.npy')
        np.save(gd_save_path, points)
