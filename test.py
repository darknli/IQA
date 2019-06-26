import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
from PIL import Image
import os
from generate_rank_data import Distortion


def get_diff(img):
    w, h = img.shape[:2]
    diff = 0.0
    count = 0
    for i in range(1, w-1):
        for j in range(1, h-1):
            count += 1
            mini_diff = 0
            if img[i, j] == img[i-1][j]:
                mini_diff += 1
            if img[i, j] == img[i][j-1]:
                mini_diff += 1
            if img[i, j] == img[i-1][j-1]:
                mini_diff += 1
            if img[i, j] == img[i][j+1]:
                mini_diff += 1
            if img[i, j] == img[i+1][j]:
                mini_diff +=1
            if img[i, j] == img[i+1][j+1]:
                mini_diff += 1
            if img[i, j] == img[i-1][j+1]:
                mini_diff += 1
            if img[i, j] == img[i+1][j-1]:
                mini_diff += 1
            diff += mini_diff/8.0
    return diff/count

def neighbor(img, kernel_size=5):
    img = img.astype(np.float)
    w, h = img.shape[:2]
    mse = []
    for i in range(0, w-kernel_size):
        for j in range(0, h-kernel_size):
            mse.append(get_diff(img[i:i+kernel_size, j:j+kernel_size]))
    return 1-np.mean(mse)

#
# for file in glob.glob(r'C:\Users\Darkn\Desktop\1\*'):
#     img = cv2.imread(file, 0)
#     print(file, neighbor(img))

def rgb2ind(img, level):
    cv2.imshow('rgb2ind.png', img)
    img = Image.open('rgb2ind.png')
    indexed = np.array(img)

    palette = img.getpalette()
    num_colors = level//3
    max_val = float(np.iinfo(img.dtype).max)
    map = np.array(palette).reshape(num_colors, 3)/max_val
    return indexed, map

def ca(img, level, return_uint8=True):
    hsize = 3
    r = img[:, :, 0]
    b = img[:, :, 2]
    r2 = r.copy()
    b2 = b.copy()
    r2[:, level:] = r[:, 1:-level+1]
    b2[:, level//2:] = b[:, 1:-level//2+1]
    img[:, :, 0] = r2
    img[:, :, 2] = b2
    img = cv2.GaussianBlur(img, (hsize, hsize), hsize/6)
    if return_uint8:
        return img.astype(np.uint8)
    else:
        return img



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
# # img = cv2.GaussianBlur(img, (73, 73), 1.5)
# # cv2.imwrite(r"C:\Users\Darkn\Desktop\1\3.jpg", img, (cv2.IMWRITE_JPEG_QUALITY, 100))
# # img_1 = cv2.imread("test.jp2")
# # cv2.imshow("1",img_1)
# cv2.waitKey()

dis = Distortion()
img = cv2.imread(r'C:\Users\Darkn\Desktop\1\1.jpg')
cv2.imshow("img", img)
cv2.moveWindow("img", 500, 500)
for i in range(4):
    new = dis.ca(img, i)
    w, h,c = new.shape
    cv2.imshow("new_%d"%i, new)
    cv2.moveWindow("new_%d"%i, i*h, 100)
cv2.waitKey()
