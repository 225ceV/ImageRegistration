
import cv2
import numpy as np
from affine_ransac import Ransac
from affine_transform import Affine
import os
from img_dehaze import *
from warp import *

# The ration of the best match over second best match
#      distance of best match
# ------------------------------- <= MATCH_RATIO
#  distance of second best match
RATIO = 0.8
import time

class Align_homo():

    def __init__(self, source_path, target_path,
                 K=3, threshold=8.0):


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
        sift = cv2.xfeatures2d.SIFT_create()
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

    def homo_matrix(self, kp_s, kp_t, fit_pos):

        # Extract corresponding points from all key points
        kp_s = kp_s[:, fit_pos[:, 0]]
        kp_t = kp_t[:, fit_pos[:, 1]]
        kp_s = kp_s.T
        kp_t = kp_t.T
        M, mask = cv2.findHomography(kp_s.reshape(-1, 1, 2), kp_t.reshape(-1, 1, 2), cv2.RANSAC,
                                     ransacReprojThreshold=self.threshold, maxIters=2000)
        mask = mask.ravel().tolist()
        return M, mask

    def warp_image(self, source, target, M):
        # # Obtain the size of target image
        # if len(target.shape) == 3:
        #     rows, cols, _ = target.shape
        # else:
        #     rows, cols = target.shape
        # # Warp the source image
        # # warp = cv2.warpAffine(source, M, (cols, rows))
        # warp = cv2.warpPerspective(source, M, (cols, rows))
        # # Merge warped image with target image to display
        # merge = np.uint8(target * 0.5 + warp * 0.5)
        #
        # img_id = os.path.split(self.source_path)[0]
        # img_id = eval(os.path.split(img_id)[1])
        # fusion_path = 'fusion'
        # cv2.imwrite(os.path.join(fusion_path, f'out_{img_id}.jpg'), merge)
        """
        source: right
        target: left

        """
        # 求出右图像的透视变化顶点

        warp_point = warp_corner(M, source)
        # 求出右图像的透视变化图像
        # imagewarp = cv2.warpPerspective(source, M, (target.shape[1] + source.shape[1], target.shape[0] + target.shape[1]))
        imagewarp = cv2.warpPerspective(source, M, (int(target.shape[1] * 1.5), int(target.shape[0] * 1.5)))    # size w, h
        # 对左右图像进行拼接，返回最后的拼接图像
        merge = Seam_Left_Right(target, imagewarp, M, warp_point, with_optim_mask=False)
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

        ## 去雾
        self.tik = time.time()
        img_source = self.dehazed(img_src)
        img_target = self.dehazed(img_tgt)
        self.tok = time.time()
        dehaze_time = self.tok - self.tik

        ## 特征提取
        self.tik = time.time()
        kp_s, desc_s, list_kp_s = self.extract_SIFT(img_source)
        kp_t, desc_t, list_kp_t = self.extract_SIFT(img_target)
        self.tok = time.time()
        extract_time = self.tok - self.tik
        # 特征点匹配  对应索引array，[[最近的点]]
        self.tik = time.time()
        fit_pos, good_matches = self.match_SIFT(desc_s, desc_t)
        self.tok = time.time()
        match_time = self.tok - self.tik
        # 计算单应性矩阵
        self.tik = time.time()
        M, mask = self.homo_matrix(kp_s, kp_t, fit_pos)
        self.tok = time.time()
        ransac_time = self.tok-self.tik
        # 融合图
        self.warp_image(img_src, img_tgt, M)   # imwrite
        # 特征点图
        out_img = cv2.drawMatches(img_src, list_kp_s, img_tgt, list_kp_t, good_matches, None, matchesMask=mask,
                                     flags=2)
        img_id = os.path.split(self.source_path)[0]
        img_id = eval(os.path.split(img_id)[1])
        out_dir = 'out'
        cv2.imwrite(os.path.join(out_dir, f'out_{img_id}.jpg'), out_img)
       # 记录矩阵
        with open('result.txt', "a+") as f:
            line = f'{M[0, 0]} {M[0, 1]} {M[0, 2]} {M[1, 0]} {M[1, 1]} {M[1, 2]} {M[2, 0]} {M[2, 1]} {M[2, 2]}\n'
            f.write(line)

        _, counts = np.unique(np.array(mask), return_counts=True)
        return dehaze_time, extract_time, match_time, ransac_time, counts[1], counts[0]+counts[1]
