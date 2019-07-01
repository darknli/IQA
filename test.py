import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
from PIL import Image
import os
from generate_rank_data import Distortion


def add_haze(image, level):
    image = image.astype(np.float)
    span = 255/level
    end = 0
    for i in range(1, level):
        begin = end
        end += span
        constant = np.ceil(end) * np.ones_like(image)
        image = np.where((image>begin) & (image<end), constant, image)
    # constant = 255 * np.ones_like(image)
    # image = np.where(img > end, constant, image)
    return image.astype(np.uint8)

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
img = cv2.imread(r'E:\Data\nsfw\porn\Dctzt-MUW4-AA-T-C.jpg')
cv2.imshow("img", img)
cv2.moveWindow("img", 500, 500)
# import time
# t = time.time()

for i in range(4):
    new = dis.quantization_noise(img, i)
    w, h,c = new.shape
    cv2.imshow("new_%d"%i, new)
    cv2.moveWindow("new_%d"%i, i*h, 100)
cv2.waitKey()


# for file in glob.glob(r'D:\temp_data\iqa\train\origin\*'):
#     try:
#         w, h, c=cv2.imread(file).shape
#         if w <=256 or h <=256:
#             raise BaseException('shan')
#     except BaseException:
#         print('removing %s'%file)
#         os.remove(file)


# img = cv2.imread(r'E:\Data\nsfw\porn\Dctzt-MUW4-AA-T-C.jpg')
# img_ = add_haze(img, 40)
# print(img_.shape)
# img__ = add_haze(img, 30)
# cv2.imshow('Source image',img)
# cv2.imshow('blur image',img_)
# cv2.imshow('blur image+',img__)
# cv2.waitKey()