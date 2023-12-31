# Created by Qixun Qu
# quqixun@gmail.com
# 2017/04/11
#


import cv2
import numpy as np
from affine_ransac import Ransac
from affine_transform import Affine
import os
from img_dehaze import *
# The ration of the best match over second best match
#      distance of best match
# ------------------------------- <= MATCH_RATIO
#  distance of second best match
RATIO = 0.8
import time

class Align():

    def __init__(self, source_path, target_path,
                 K=3, threshold=1.0):


        self.source_path = source_path
        self.target_path = target_path
        self.K = K
        self.threshold = threshold
        self.tik = None
        self.tok = None

    def read_image(self, path, mode=1):
        return cv2.imread(path, mode)

    def extract_SIFT(self, img):

        # Convert the image to grayscale
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
        # Extract key points and SIFT descriptors
        sift = cv2.SIFT_create()
        list_kp, desc = sift.detectAndCompute(img_gray, None)

        # Extract positions of key points
        kp = np.array([p.pt for p in list_kp]).T

        return kp, desc, list_kp

    def match_SIFT(self, desc_s, desc_t):
        # Match descriptor and obtain two best matches
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc_s, desc_t, k=2)

        # Initialize output variable
        fit_pos = np.array([], dtype=np.int32).reshape((0, 2))

        matches_num = len(matches)
        good_match = []
        ## 寻找好的匹配点

        for m, n in matches:
            if m.distance < RATIO * n.distance:
                temp = np.array([m.queryIdx, m.trainIdx])
                fit_pos = np.vstack((fit_pos, temp))
                # good_match.append((m, n))
                good_match.append(m)

        return fit_pos, good_match

    def affine_matrix(self, kp_s, kp_t, fit_pos):
        # Extract corresponding points from all key points
        kp_s = kp_s[:, fit_pos[:, 0]]
        kp_t = kp_t[:, fit_pos[:, 1]]

        # Apply RANSAC to find most inliers
        _, _, inliers, mask = Ransac(self.K, self.threshold).ransac_fit(kp_s, kp_t)

        # Extract all inliers from all key points
        kp_s = kp_s[:, inliers[0]]
        kp_t = kp_t[:, inliers[0]]

        # Use all inliers to estimate transform matrix
        A, t = Affine().estimate_affine(kp_s, kp_t)
        M = np.hstack((A, t))


        return M, mask

    def warp_image(self, source, target, M):
        # Obtain the size of target image
        if len(target.shape) == 3:
            rows, cols, _ = target.shape
        else:
            rows, cols = target.shape
        # Warp the source image
        warp = cv2.warpAffine(source, M, (cols, rows))

        # Merge warped image with target image to display
        merge = np.uint8(target * 0.5 + warp * 0.5)

        # Show the result
        # cv2.imshow('img', merge)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        img_id = os.path.split(self.source_path)[0]
        img_id = eval(os.path.split(img_id)[1])
        fusion_path = 'fusion'
        cv2.imwrite(os.path.join(fusion_path, f'out_{img_id}.jpg'), merge)

        return

    def dehazed(self, img):
        return dehaze(img)  # float64

    def align_image(self):
        '''
        提取匹配特征点，计算单应性矩阵，并将结果做储存
        :return: 处理时间，inner_points占比
        '''
        img_src = self.read_image(self.source_path)
        img_tgt = self.read_image(self.target_path)
        self.tik = time.time()
        ## 去雾
        # img_source = self.dehazed(img_src)
        # img_target = self.dehazed(img_tgt)
        img_source = img_src
        img_target = img_tgt
        ## 边缘提取，和描述
        # img_source = cv2.Canny(img_src, 100, 200)
        # img_target = cv2.Canny(img_tgt, 100, 200)
        kp_s, desc_s, list_kp_s = self.extract_SIFT(img_source)
        kp_t, desc_t, list_kp_t = self.extract_SIFT(img_target)
        # 特征点匹配  对应索引array，[[最近的点]]
        fit_pos, good_matches = self.match_SIFT(desc_s, desc_t)
        # 计算单应性矩阵
        M, mask = self.affine_matrix(kp_s, kp_t, fit_pos)
        self.tok = time.time()
        registration_time = self.tok-self.tik
        # 融合图
        self.warp_image(img_source, img_target, M)   # imwrite
        # 特征点图
        out_img = cv2.drawMatches(img_source, list_kp_s, img_target, list_kp_t, good_matches, None, matchesMask=mask,
                                     flags=2)
        img_id = os.path.split(self.source_path)[0]
        img_id = eval(os.path.split(img_id)[1])
        out_dir = 'out'
        cv2.imwrite(os.path.join(out_dir, f'out_{img_id}.jpg'), out_img)
       # 记录矩阵
        with open('result.txt', "a+") as f:
            line = f'{M[0, 0]} {M[0, 1]} {M[0, 2]} {M[1, 0]} {M[1, 1]} {M[1, 2]} 0 0 1\n'
            f.write(line)

        self.tok = time.time()
        total_time = self.tok - self.tik
        _, counts = np.unique(np.array(mask), return_counts=True)
        return registration_time, total_time, counts[1], counts[0]+counts[1]

if __name__ == '__main__':
    # source_path = 'data/0/33_B.jpg'
    # target_path = 'data/0/33_A.jpg'
    # pathes = [os.path.join('data', i) for i in ['33', '75', '95']]  # 重点图片
    pathes = [os.path.join('data', str(i)) for i in range(100)]    # 全部图片
    registration_time_list = []
    total_time_list = []
    for path in pathes:
        source_path = os.path.join(path, 'B.jpg')
        target_path = os.path.join(path, 'A.jpg')
        print(f'正在处理图像路径：{path}')
        # Create instance
        al = Align(source_path, target_path, threshold=1)
        # Image transformation
        # al.align_image()
        try:
            registration_time, total_time, _, _= al.align_image()
            registration_time_list.append(registration_time)
            total_time_list.append(total_time)
        except:
            # 33 75 95无法检测
            print("算法无法检测")
            with open('result.txt', "a+") as f:
                line = f'该图片提取特征点数量不够\n'
                f.write(line)
    print(f'总用时{sum(total_time_list):.3f}s, 配准净时长{sum(registration_time_list):.3f}s')

