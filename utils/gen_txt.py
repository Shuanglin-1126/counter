import os
import re
import numpy as np


log_path = r'/data/che_xiao/crowd_count/counter/checkpoints/test.log'
out_path = r'/data/che_xiao/crowd_count/counter/checkpoints/test.txt'


with open(log_path, mode="rt", encoding="utf-8") as file, open(out_path, "w") as wrt:
    lines = file.readlines()
    for line in lines:
        if 'name' in line:
            data = re.findall(r'\d+',line)
            name_d = int(data[-3])
            count_1 = np.array(data[-2], dtype=float)
            count_2 = np.array(data[-1], dtype=float)
            count = np.array((count_1 + 0.01 * count_2),dtype=float)
            str_format = r'{:06d},{:.2f}'.format(name_d, count)
            wrt.write(str_format + '\n')
