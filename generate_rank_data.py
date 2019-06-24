"""
    wn_level = [0.001, 0.005,0.01, 0.05]; % #1 Gaussian noise
    gnc_level = [0.0140,0.0198,0.0343,0.0524];     % #2 Gaussian noise in color components
    hfn_level = [0.001,0.005,0.01,0.05];      % #5 High frequency noise
    in_level = [0.005,0.01,0.05,0.1];     % #6 Impulse noise
    qn_level = int32([255./27,255./39,255./55,255./76]);  % #7 Quantization noise
    gblur_level = [7,15,39,91];  % #8 Gaussian Blur
    id_level = [0.001, 0.005,0.01, 0.05];              % #9 Image Denoising
    jpeg_level = [43,12,7,4];  % #10 JPEG compression
    jp2k_level = [0.46,0.16,0.07,0.04]; % #11  JP2K compression
    nepn_level = [30,70,150,300];  % #14  Non eccentricity pattern noise
    bw_level = [2,4,8,16,32];    % #15  Local block-wise distortions of different intensity
    ms_level = [15,30,45,60] ;  % #16  Mean shift MSH =[15,30,45,60] MSL = [-15,-30,-45,-60]
    cc_level = [0.85,0.7,0.55,0.4];  % #17  Contrast change [0.85,0.7,0.55,0.4] [1.2,1.4,1.6,1.8]
    cs_level = [0.4,0,-0.4,-0.8];  % #18  color saturation
    mgn_level = [0.05, 0.09,0.13, 0.2];   % #19  Multiplicative Gaussian noise
    cqd_level = [64,32, 16,8,4];   % #22  Color quantization dither
    ca_level = [2,6,10,14];   % #23  Color aberrations
"""
import numpy as np
import cv2
import os

class Distortion:
    def __init__(self):
        self.wn_level = [0.001, 0.005,0.01, 0.05]
        self.gnc_level = [0.0140,0.0198,0.0343,0.0524]
        self.hfn_level = [0.001,0.005,0.01,0.05]
        self.in_level = [0.005,0.01,0.05,0.1]
        self.qn_level = [255.//27,255.//39,255.//55,255.//76]
        self.gblur_level = [7,15,39,91]
        self.id_level = [0.001, 0.005,0.01, 0.05]
        self.jpeg_level = [43,12,7,4]
        self.jp2k_level = [0.46,0.16,0.07,0.04]
        self.nepn_level = [30,70,150,300]
        self.bw_level = [2, 4, 8, 16, 32]
        self.ms_level = [15,30,45,60]
        self.cc_level = [0.85,0.7,0.55,0.4]
        self.mgn_level = [0.05, 0.09,0.13, 0.2]
        self.cqd_level = [64,32, 16,8,4]
        self.ca_level = [2,6,10,14]

    def gaussian_noise(self, img, level, is_rgb=False):
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
            noise = np.random.normal(0, self.wn_level[level], shape)
        else:
            noise = np.random.normal(0, self.gnc_level[level], shape)
        img += noise
        img *= 255
        img = np.clip(img, 0, 255)
        return img

    def gassian_blur(self, img, level, sigma=0.1):
        """
        高斯模糊
        :param img: 输入图像rgb矩阵
        :param level: 卷积核大小等级
        :param sigma: sigma大小
        """
        img = cv2.GaussianBlur(img, self.gblur_level[level], sigma)
        return img

    def impulse_noise(self, img, level, color=None):
        """
        脉冲噪声（椒盐噪声），随机把像素点变成0或255
        :param img: 输入图像rgb矩阵
        :param level: 噪声比例等级
        """
        salt_prob = np.random.random(0, self.in_level[level])
        salt_noise = np.zeros_like(img)
        pepper_noise = 255 * np.ones_like(img)
        prob_mat = np.random.uniform(0, 1, img.shape[:2])
        pepper_img = np.where(prob_mat<self.in_level[level], pepper_noise, img)
        sp_img = np.where(pepper_img<salt_prob, salt_noise, img)
        return sp_img

    def quantization_noise(self, img, level):
        """
        otsu多级图像阈值分割（未完成）
        :param img: 输入图像rgb矩阵
        :param level: 分割等级
        :return:
        """
        ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th2

    def jpeg_compression(self, img, level):
        """
        jpeg压缩
        :param img: 输入图像rgb矩阵
        :param level: 压缩等级
        """
        cv2.imwrite("jpeg_temp.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_level[level]])
        img = cv2.imread("jpeg_temp.jpg")
        os.remove("jpeg_temp.jpg")
        return img

    def nepn(self, img, level):
        w, h = img.shape[:2]
        nepn_img = img
        print(nepn_img.shape, img.shape)
        for _ in range(self.nepn_level[level]):
            rw = np.random.randint(1, w - 30)
            rh = np.random.randint(1, h - 30)
            origin_rw = rw + 5
            origin_rh = rh + 5
            nepn_img[rw:rw + 14, rh:rh + 14, :] = img[origin_rw:origin_rw + 14, origin_rh:origin_rh + 14, :]

        return nepn_img