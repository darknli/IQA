import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
from PIL import Image
import os
from generate_rank_data import Distortion


def add_haze(image, t=0.6, A=1):
    '''
        添加雾霾
        t : 透视率 0~1
        A : 大气光照
    '''
    image = image.astype(np.float)
    constant = 50 * np.ones_like(images)
    index = np.where(image> 25 and img <50, )
    return image

def gaussian_noise(img, level, is_rgb=True, return_uint8=True):
    """
    高斯噪声
    :param img: 输入图像rgb矩阵
    :param level: 方差等级
    :param is_rgb: 灰度或彩色
    """
    img /=255.0
    shape = img.shape
    if not is_rgb:
        shape[-1] = 1
        noise = np.random.normal(0, level, shape)
    else:
        noise = np.random.normal(0, level, shape)
    img += noise
    img *= 255
    img = np.clip(img, 0, 255)
    if return_uint8:
        return img.astype(np.uint8)
    else:
        return img
# name = "ca"
# img = cv2.imread(r'C:\Users\Darkn\Desktop\1\1.jpg')
# img = eval(name)(img, 8)
# cv2.imshow("测试", img)
# img = cv2.GaussianBlur(img, (73, 73), 1.5)
# cv2.imwrite(r"C:\Users\Darkn\Desktop\1\3.jpg", img, (cv2.IMWRITE_JPEG_QUALITY, 100))
# img_1 = cv2.imread("test.jp2")
# cv2.imshow("1",img_1)
# cv2.waitKey()

dis = Distortion()
# img = cv2.imread(r'C:\Users\Darkn\Desktop\1\1.jpg')
# cv2.imshow("img", img)
# cv2.moveWindow("img", 500, 500)
# import time
# t = time.time()
# i1 = dis.hf_noise(img, 1, 0)
# print(time.time()-t)
# cv2.imshow('i1', i1)
#
# i2 = dis.motion_blur(img, 100)
# cv2.imshow('i2', i2)
# for i in range(4):
#     new = dis.ca(img, i)
#     w, h,c = new.shape
#     cv2.imshow("new_%d"%i, new)
#     cv2.moveWindow("new_%d"%i, i*h, 100)
# cv2.waitKey()


# for file in glob.glob(r'D:\temp_data\iqa\train\origin\*'):
#     try:
#         w, h, c=cv2.imread(file).shape
#         if w <=256 or h <=256:
#             raise BaseException('shan')
#     except BaseException:
#         print('removing %s'%file)
#         os.remove(file)


img = cv2.imread(r'C:\Users\Darkn\Desktop\1\1.jpg')
img_ = add_haze(img)
print(img_.shape)
# img__ = dis.motion_blur(img, 40, angle=45)
cv2.imshow('Source image',img)
cv2.imshow('blur image',img_)
# cv2.imshow('blur image+',img__)
cv2.waitKey()