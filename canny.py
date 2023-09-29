import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from img_dehaze import *

src = cv.imread('Images/33_A.jpg')
src = dehaze(src)
src_edges = cv.Canny(src, 150, 200)
tar = cv.imread('Images/33_B.jpg')
tar = dehaze(tar)
tar_edges = cv.Canny(tar, 150, 200)

plt.subplot(221), plt.imshow(src, cmap='gray')
plt.title('Original src'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(src_edges, cmap='gray')
plt.title('Edge src'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(tar, cmap='gray')
plt.title('Original tar'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(tar_edges, cmap='gray')
plt.title('Edge tar'), plt.xticks([]), plt.yticks([])


plt.show()
