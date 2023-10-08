import numpy as np
import pandas as pd

from align_transform import Align
import os
import shutil
from dehaze_sift_homo import Align_homo as d_sift_homo
from dehaze_surf_homo import Align_homo as d_surf_homo
from sift_homo import Align_homo as SIFT_homo
from dehaze_orb import Align_homo as orb_homo
from sift_affine import Align_homo as SIFT_affine

"""
data
"""
# 全部图片
pathes = [os.path.join("data1", str(i)) for i in range(100)]
# 重点图片
# pathes = [os.path.join('data', i) for i in ['1', '24', '28', '32', '33', '34', '54', '56','69', '75', '84', '89', '95']]


def clear_last():
    if "result.txt" in os.listdir():
        os.remove("result.txt")
    if "out" in os.listdir():
        shutil.rmtree("out")
    if "fusion" in os.listdir():
        shutil.rmtree("fusion")

    return None


"""
pipline
"""
clear_last()
if "out" not in os.listdir():
    os.mkdir("out")
if "fusion" not in os.listdir():
    os.mkdir("fusion")
if "result.txt" not in os.listdir():
    with open("result.txt", "w") as f:
        pass

dehaze_time_list = []
extract_time_list = []
match_time_list = []
ransac_time_list = []
total_time_list = []
ratios = []
inners = []
for path in pathes:
    print(f"正在处理图像路径：{path}")
    src_path = os.path.join(path, "A.jpg")
    tar_path = os.path.join(path, "B.jpg")
    al = SIFT_affine(src_path, tar_path, threshold=1.0)
    # al = d_sift_homo(src_path, tar_path, threshold=1.0)
    # al = d_surf_homo(src_path, tar_path, threshold=1.0)
    # al = orb_homo(src_path, tar_path, threshold=1.0)
    dehaze_time, extract_time, match_time, ransac_time, inner, good = al.align_image()
    # try:
    #     dehaze_time, extract_time, match_time, ransac_time, inner, good = al.align_image()
    # except:
    #     print('无法处理当前图片')
    #     continue
    ratio = inner / good
    dehaze_time_list.append(dehaze_time)
    extract_time_list.append(extract_time)
    match_time_list.append(match_time)
    ransac_time_list.append(ransac_time)
    total_time_list.append(sum((dehaze_time, extract_time, match_time, ransac_time)))
    ratios.append(ratio)
    inners.append(inner)
    print(f"当前图像inner点数量{inner}, 占比{ratio:.3f}")
print(
    f"总用时{sum(total_time_list):.3f}s\n",
    f"平均inner点数量{sum(inners)/len(inners):.3f}",
    f"平均inner_points比例{sum(ratios)/len(ratios):.3f}",
)
few = np.where(np.array(inners) < 100)
print(f"以下图像inner点较少")
for i in few[0]:
    print(f"图像{i}: inners数量{inners[i]}")

log_data = np.array([dehaze_time_list, extract_time_list, match_time_list, ransac_time_list, total_time_list, inners, ratios])
log = pd.DataFrame(data=log_data, index=['dehaze_time', 'extract_time', 'match_time', 'ransac_time', 'totol_time', 'inlier', 'ratios'])
log.to_csv('log.csv')
