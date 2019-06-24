import cv2
import matplotlib.pyplot as plt
import numpy as np


def nepn(img, level):
    w, h = img.shape[:2]
    nepn_img = img
    print(nepn_img.shape, img.shape)
    for _ in range(level):
        rw = np.random.randint(1, w - 30)
        rh = np.random.randint(1, h - 30)
        origin_rw = rw+5
        origin_rh = rh+5
        nepn_img[rw:rw+14, rh:rh+14, :] = img[origin_rw:origin_rw+14, origin_rh:origin_rh+14, :]

    return nepn_img


img = cv2.imread(r'C:\Users\Darkn\Desktop\profile\10.jpg')
cv2.imwrite("test.jpg", nepn(img, 90))
# img_1 = cv2.imread("test.jp2")
# cv2.imshow("1",img_1)
# cv2.waitKey()