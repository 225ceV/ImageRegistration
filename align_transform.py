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
                 K=3, threshold=1):
        ''' __INIT__

            Initialize the instance.

            Input arguments:

            - source_path : the path of sorce image that to be warped
            - target_path : the path of target image
            - K : the number of corresponding points, default is 3
            - threshold : a threshold determins which points are outliers
            in the RANSAC process, if the residual is larger than threshold,
            it can be regarded as outliers, default value is 1

        '''

        self.source_path = source_path
        self.target_path = target_path
        self.K = K
        self.threshold = threshold
        self.tik = None
        self.tok = None

    def read_image(self, path, mode=1):
        ''' READ_IMAGE

            Load image from file path.

            Input arguments:

            - path : the image to be read
            - mode : 1 for reading color image, 0 for grayscale image
            default is 1

            Output:

            - the image to be processed

        '''

        return cv2.imread(path, mode)

    def extract_SIFT(self, img):
        ''' EXTRACT_SIFT

            Extract SIFT descriptors from the given image.

            Input argument:

            - img : the image to be processed

            Output:

            -kp : positions of key points where descriptors are extracted
            - desc : all SIFT descriptors of the image, its dimension
            will be n by 128 where n is the number of key points


        '''

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
        ''' MATCH_SIFT

            Match SIFT descriptors of source image and target image.
            Obtain the index of conrresponding points to do estimation
            of affine transformation.

            Input arguments:

            - desc_s : descriptors of source image
            - desc_t : descriptors of target image

            Output:

            - fit_pos : index of corresponding points

        '''

        # Match descriptor and obtain two best matches
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc_s, desc_t, k=2)

        # Initialize output variable
        fit_pos = np.array([], dtype=np.int32).reshape((0, 2))

        matches_num = len(matches)
        good_match = []
        ## 寻找好的匹配点
        for i in range(matches_num):
            # Obtain the good match if the ration id smaller than 0.8
            if matches[i][0].distance <= RATIO * matches[i][1].distance:
                temp = np.array([matches[i][0].queryIdx,
                                 matches[i][0].trainIdx])
                # Put points index of good match
                fit_pos = np.vstack((fit_pos, temp))
                good_match.append(matches[i])

        """
        test
        """
        # d1 = np.array([good_match[i][0].distance for i in range(15)])
        # d2 = np.array([good_match[i][1].distance for i in range(15)])
        # print(d1/d2)
        # print(fit_pos, fit_pos.shape)
        return fit_pos, good_match

    def affine_matrix(self, kp_s, kp_t, fit_pos):
        ''' AFFINE_MATRIX

            Compute affine transformation matrix by corresponding points.

            Input arguments:

            - kp_s : key points from source image
            - kp_t : key points from target image
            - fit_pos : index of corresponding points

            Output:

            - M : the affine transformation matrix whose dimension
            is 2 by 3

        '''

        # Extract corresponding points from all key points
        kp_s = kp_s[:, fit_pos[:, 0]]
        kp_t = kp_t[:, fit_pos[:, 1]]

        # Apply RANSAC to find most inliers
        _, _, inliers = Ransac(self.K, self.threshold).ransac_fit(kp_s, kp_t)

        # Extract all inliers from all key points
        kp_s = kp_s[:, inliers[0]]
        kp_t = kp_t[:, inliers[0]]

        # Use all inliers to estimate transform matrix
        A, t = Affine().estimate_affine(kp_s, kp_t)
        M = np.hstack((A, t))


        return M

    def warp_image(self, source, target, M):
        ''' WARP_IMAGE

            Warp the source image into target with the affine
            transformation matrix.

            Input arguments:

            - source : the source image to be warped
            - target : the target image
            - M : the affine transformation matrix

        '''

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
        ''' ALIGN_IMAGE

            Warp the source image into target image.
            Two images' path are provided when the
            instance Align() is created.

        '''
        # Load source image and target image
        img_src = self.read_image(self.source_path)
        img_tgt = self.read_image(self.target_path)
        self.tik = time.time()
        ## 去雾
        img_src = self.dehazed(img_src)
        img_tgt = self.dehazed(img_tgt)
        ## 边缘提取
        img_source = cv2.Canny(img_src, 100, 200)
        img_target = cv2.Canny(img_tgt, 100, 200)
        # Extract key points and SIFT descriptors from
        # source image and target image respectively
        kp_s, desc_s, list_kp_s = self.extract_SIFT(img_source)
        kp_t, desc_t, list_kp_t = self.extract_SIFT(img_target)

        # Obtain the index of correcponding points
        fit_pos, good_matches = self.match_SIFT(desc_s, desc_t)
        # good_matches.sort(key=lambda x : x[0].distance/x[1].distance)     # 合理性存疑

        # Compute the affine transformation matrix
        M = self.affine_matrix(kp_s, kp_t, fit_pos)
        self.tok = time.time()
        registration_time = self.tok-self.tik
        # print(f"Registration Time: {self.tok-self.tik:.4f}s")
        # # Warp the source image and display result
        out_img = cv2.drawMatchesKnn(img_source, list_kp_s, img_target, list_kp_t, good_matches[:100], None, flags=2)
        self.warp_image(img_src, img_tgt, M)   # imwrite
        # cv2.namedWindow("image", 0)
        # cv2.resizeWindow("image", 1920, 540)
        # cv2.imshow('image', out_img)  # 展示图片
        # cv2.waitKey(0)  # 等待按键按下
        # cv2.destroyAllWindows()  # 清除所有窗口
        # img_dir = os.path.split(self.source_path)[0]
        img_id = os.path.split(self.source_path)[0]
        img_id = eval(os.path.split(img_id)[1])
        out_dir = 'out'
        cv2.imwrite(os.path.join(out_dir, f'out_{img_id}.jpg'), out_img)
        with open('result.txt', "a+") as f:
            line = f'{M[0, 0]} {M[0, 1]} {M[0, 2]} {M[1, 0]} {M[1, 1]} {M[1, 2]} 0 0 1\n'
            f.write(line)

        self.tok = time.time()
        total_time = self.tok - self.tik
        return registration_time, total_time

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
            registration_time, total_time = al.align_image()
            registration_time_list.append(registration_time)
            total_time_list.append(total_time)
        except:
            # 33 75 95无法检测
            print("算法无法检测")
            with open('result.txt', "a+") as f:
                line = f'该图片提取特征点数量不够\n'
                f.write(line)
    print(f'总用时{sum(total_time_list):.3f}s, 配准净时长{sum(registration_time_list):.3f}s')

