'''代码1
先使用SIFT算法提取特征，完成图像的匹配
'''
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

#导入两张图片
imgname_01 = 'Images/mona_source.png'
imgname_02 = 'Images/mona_target.jpg'
#利用现有的cv2模块方法，创建一个SIFT的对象
sift = cv2.SIFT_create()


# BFmatcher（Brute-Force Matching）暴力匹配   暴力方法找到点集1中每个descriptor在点集2中距离最近的descriptor；找寻到的距离最小就认为匹配

#应用BFMatch暴力方法找到点集1中每个descriptor在点集2中距离最近的descriptor；找寻到的距离最小就认为匹配er.knnMatch( )函数来进行核心的匹配，knnMatch（k-nearest neighbor classification）k近邻分类算法。
# 进行特征检测，得到2张图片的特征点和描述子

img_01 = cv2.imread(imgname_01)
img_02 = cv2.imread(imgname_02)
keypoint_01, descriptor_01 = sift.detectAndCompute(img_01, None)
keypoint_02, descriptor_02 = sift.detectAndCompute(img_02, None)


bf = cv2.BFMatcher()#默认是欧氏距离 cv2.NORM_L2
# k = 2 返回点集1中每个描述点在点集2中 距离最近的2个匹配点
matches = bf.knnMatch(descriptor_01, descriptor_02, k = 2)

print(matches[0][0])
# 调整ratio
ratio = 0.8
good = []

#  m n 相比较各自的距离
for m,n in matches:
    #第一个m匹配的是最近邻，第二个n匹配的是次近邻。直觉上，一个正确的匹配会更接近第一个邻居。
    if m.distance < ratio * n.distance:
        good.append([m])
img5 = cv2.drawMatchesKnn(img_01, keypoint_01, img_02, keypoint_02, good, None, flags=2)

img_sift = cv2.cvtColor(img5, cv2.COLOR_BGR2RGB) #灰度处理图像
cv2.imshow('img', img_sift)
cv2.waitKey(0)  # 等待按键按下

cv2.destroyAllWindows()