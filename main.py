from align_transform import Align
import os
import shutil

"""
data
"""
# 全部图片
pathes = [os.path.join('data', str(i)) for i in range(100)]
# 重点图片
# pathes = [os.path.join('data', i) for i in ['33', '75', '95']]

def clear_last():
    if 'result.txt' in os.listdir():
        os.remove('result.txt')
    if 'out' in os.listdir():
        shutil.rmtree('out')
    if 'fusion' in os.listdir():
        shutil.rmtree('fusion')

    return None

"""
pipline
"""
clear_last()
if 'out' not in os.listdir():
    os.mkdir('out')
if 'fusion' not in os.listdir():
    os.mkdir('fusion')
if 'result.txt' not in os.listdir():
    with open('result.txt', 'w') as f:
        pass

registration_time_list = []
total_time_list = []
for path in pathes:
    print(f'正在处理图像路径：{path}')
    src_path = os.path.join(path, 'B.jpg')
    tar_path = os.path.join(path, 'A.jpg')
    al = Align(src_path, tar_path, threshold=1)
    registration_time, total_time = al.align_image()
    registration_time_list.append(registration_time)
    total_time_list.append(total_time)
print(f'总用时{sum(total_time_list):.3f}s, 配准净时长{sum(registration_time_list):.3f}s')
