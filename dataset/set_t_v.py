import os
import shutil
import random

save_path = r'/data/che_xiao/crowd_count/外部数据集/DroneRGBT/DroneRGBT/Train'
root_path = r'/data/che_xiao/crowd_count/外部数据集/DroneRGBT/DroneRGBT/Test'

num_train = len(os.listdir(os.path.join(save_path, 'RGB')))
path_ori = os.path.join(root_path, 'RGB')
files = os.listdir(path_ori)
for file in files:
    name_ori = file.split('.')[0]
    for x in ['RGB', 'Infrared', 'GT_']:
        path_sa = os.path.join(save_path, x)
        path_or = os.path.join(root_path, x)
        if x == 'RGB':
            name_sa = str(int(name_ori) + num_train)
        else:
            name_sa = str(int(name_ori) + num_train) + 'R'
        if x == 'GT_':
            name_sa = name_sa + '.xml'
        else:
            name_sa = name_sa + '.jpg'
        if x == 'Infrared':
            file1 = file.replace('.jpg', 'R.jpg')
        elif x == 'GT_':
            file1 = file.replace('.jpg', 'R.xml')
        elif x == 'RGB':
            file1 = file
        file_save = os.path.join(path_sa, name_sa)
        shutil.move(os.path.join(path_or, file1), file_save)
