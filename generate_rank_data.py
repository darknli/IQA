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
from PIL import Image
from glob import glob
from tqdm import tqdm

class Distortion:
    def __init__(self):
        self.wn_level = [0.001, 0.005, 0.01, 0.05]
        self.gnc_level = [0.0140, 0.0198, 0.0343, 0.0524]
        self.hfn_level = [0.001, 0.005, 0.01, 0.05]
        self.in_level = [0.005, 0.01, 0.05, 0.1]
        self.qn_level = [255.//27, 255.//39, 255.//55, 255.//76]
        self.gblur_level = [7, 15, 39, 91]
        self.id_level = [0.001, 0.005, 0.01, 0.05]
        self.jpeg_level = [43, 12, 7, 4]
        self.jp2k_level = [0.46, 0.16, 0.07, 0.04]
        self.nepn_level = [30, 70, 150, 300]
        self.bw_level = [2, 4, 8, 16, 32]
        self.ms_level = [15, 30, 45, 60]
        self.cc_level = [0.85, 0.7, 0.55, 0.4]
        self.mgn_level = [0.05, 0.09, 0.13, 0.2]
        self.cqd_level = [64, 32, 16, 8, 4]
        self.ca_level = [2, 6, 10, 14]
        self.idx2func = {
            1: "gaussian_noise",
            2: "gassian_blur",
            3: "impulse_noise",
            4: "quantization_noise",
            5: "jpeg_compression",
            6: "nepn",
            7: "block_wise",
            8: "hf_noise",
            9: "multi_gn",
            10:"cqd",
            11:"ca"
        }

    def generate_data(self, data, rank_root):
        imgs = glob(os.path.join(data, '*'))
        process_length = len(imgs) * 12 * 4

        for i in range(1, 12):
            func_path = os.path.join(rank_root, str(i))
            if not os.path.exists(func_path):
                os.mkdir(func_path)
            print('generating the %s ...' % self.idx2func[i])
            with tqdm(total=process_length) as pbar:
                for level in range(4):
                    level_path = os.path.join(func_path, str(level))
                    if not os.path.exists(level_path):
                        os.mkdir(level_path)
                    for img in imgs:
                        img_name = os.path.basename(img)
                        img = cv2.imread(img).astype(np.float)
                        distorted_img = eval("self."+self.idx2func[i])(img, level)
                        save_path = os.path.join(level_path, img_name)
                        cv2.imwrite(save_path, distorted_img)
                        pbar.update(1)


    def gaussian_noise(self, img, level, is_rgb=True, return_uint8=True):
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
        if return_uint8:
            return img.astype(np.uint8)
        else:
            return img

    def gassian_blur(self, img, level, return_uint8=True):
        """
        高斯模糊
        :param img: 输入图像rgb矩阵
        :param level: 卷积核大小等级
        :param sigma: sigma大小
        """
        img = cv2.GaussianBlur(img, (self.gblur_level[level], self.gblur_level[level]), np.sqrt(self.gblur_level[level]/6))
        if return_uint8:
            return img.astype(np.uint8)
        else:
            return img

    def impulse_noise(self, img, level, color=None, return_uint8=True):
        """
        脉冲噪声（椒盐噪声），随机把像素点变成0或255
        :param img: 输入图像rgb矩阵
        :param level: 噪声比例等级
        """
        salt_prob = np.random.uniform(0, self.in_level[level])
        salt_noise = np.zeros_like(img)
        pepper_noise = 255 * np.ones_like(img)
        prob_mat = np.random.uniform(0, 1, img.shape[:2])
        pepper_img = np.where(prob_mat<self.in_level[level], pepper_noise, img)
        img = np.where(pepper_img<salt_prob, salt_noise, img)
        if return_uint8:
            return img.astype(np.uint8)
        else:
            return img

    def quantization_noise(self, img, level, return_uint8=True):
        """
        otsu多级图像阈值分割（未完成）
        :param img: 输入图像rgb矩阵
        :param level: 分割等级
        """
        ret2, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if return_uint8:
            return img.astype(np.uint8)
        else:
            return img

    def jpeg_compression(self, img, level, return_uint8=True):
        """
        jpeg压缩
        :param img: 输入图像rgb矩阵
        :param level: 压缩等级
        """
        cv2.imwrite("jpeg_temp.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_level[level]])
        img = cv2.imread("jpeg_temp.jpg")
        os.remove("jpeg_temp.jpg")
        if return_uint8:
            return img.astype(np.uint8)
        else:
            return img

    def nepn(self, img, level, return_uint8=True):
        """
        效果：在图像上生成多个“像其他图像遗留在这里”的图，效果不错
        """
        w, h = img.shape[:2]
        nepn_img = img.copy()
        print(nepn_img.shape, img.shape)
        for _ in range(self.nepn_level[level]):
            rw = np.random.randint(1, w - 30)
            rh = np.random.randint(1, h - 30)
            origin_rw = rw + 5
            origin_rh = rh + 5
            nepn_img[rw:rw + 14, rh:rh + 14, :] = img[origin_rw:origin_rw + 14, origin_rh:origin_rh + 14, :]
        if return_uint8:
            return nepn_img.astype(np.uint8)
        else:
            return nepn_img

    def block_wise(self, img, level, return_uint8=True):
        """
        效果：在图片上中生成多个随机的小方块，看起来很突兀，可以不要
        """
        w, h = img.shape[:2]

        r_mean = np.mean(img[:, :, 0])
        g_mean = np.mean(img[:, :, 1])
        b_mean = np.mean(img[:, :, 2])

        level_map = {
            5: lambda x: x,
            4: lambda x: x,
            3: lambda x: x + 30,
            2: lambda x: x + 50,
            1: lambda x: np.zeros_like(x)
        }
        g_mean = level_map[level](g_mean)
        concat = np.array([r_mean, g_mean, b_mean])
        np.tile(concat, (1, 32, 32)).transpose((2, 1, 0))
        for _ in range(self.bw_level[level]):
            rw = np.random.randint(1, w - 31)
            rh = np.random.randint(2, h - 31)
            img[rw:rw + 31, rh:rh + 31, :] = concat
        if return_uint8:
            return img.astype(np.uint8)
        else:
            return img

    def hf_noise(self, img, level, return_uint8=True):
        """
        高频噪声
        """
        def ghp(img, thresh):
            r, c = img.shape[:2]
            d0 = thresh
            d = np.zeros((r, c))
            h = np.zeros((r, c))
            for i in range(r):
                for j in range(c):
                    d[i, j] = np.sqrt((i + 1 - r / 2) ** 2 + (j + 1 - c / 2) ** 2)
            for i in range(r):
                for j in range(c):
                    h[i, j] = 1 - np.exp(-d[i, j] ** 2 / (2 * (d0 ** 2)))
            res = h * img
            return res

        img = img.astype(np.float) / 255
        img_fft = np.fft.fft2(img)
        thresh = 10
        ghp1 = np.expand_dims(ghp(img_fft[:, :, 0], thresh), axis=-1)
        ghp2 = np.expand_dims(ghp(img_fft[:, :, 1], thresh), axis=-1)
        ghp3 = np.expand_dims(ghp(img_fft[:, :, 2], thresh), axis=-1)

        ifft2 = np.fft.ifft2(np.concatenate([ghp1, ghp2, ghp3], axis=-1))
        img = np.real(ifft2)
        img = np.clip(255 * img, 0, 255)
        img = self.gaussian_noise(img, self.hfn_level[level])
        if return_uint8:
            return img.astype(np.uint8)
        else:
            return img

    def img_denoising(self, img, level, return_uint8=True):
        """
        未完成
        """
        distorted_img = self.gaussian_noise(img, level)
        yrgb = distorted_img.astype(np.float) / 255.
        sigma = 25
        zrgb = yrgb + (25.0 / 255) * np.random.normal(0, 1, yrgb.shape)
        if return_uint8:
            return zrgb.astype(np.uint8)
        else:
            return zrgb

    def multi_gn(self, img, level, return_uint8=True):
        img = img.astype(np.float)
        noise_only_img = 1 + np.random.normal(0, self.mgn_level[level], img.shape[:2])
        img[:, :, 0] *= noise_only_img
        noise_only_img = 1 + np.random.normal(0, self.mgn_level[level], img.shape[:2])
        img[:, :, 1] *= noise_only_img
        noise_only_img = 1 + np.random.normal(0, self.mgn_level[level], img.shape[:2])
        img[:, :, 2] *= noise_only_img
        img = np.clip(img, 0, 255)
        if return_uint8:
            return img.astype(np.uint8)
        else:
            return img

    def cqd(self, img, level, return_uint8=True):
        """
        Color quantization dither
        """
        cv2.imwrite("cqd.png", img)
        lena = Image.open("cqd.png")
        lena_P_dither = lena.convert("P", palette=Image.ADAPTIVE, colors=self.cqd_level[level])
        img = lena_P_dither.convert("RGB")
        img.save("cqd_1.png")
        img = cv2.imread("cqd_1.png")
        del lena, lena_P_dither
        os.remove("cqd.png")
        os.remove("cqd_1.png")
        if return_uint8:
            return img
        else:
            return img.astype(np.float)

    def ca(self, img, level, return_uint8=True):
        hsize = 3
        r = img[:, :, 0]
        b = img[:, :, 2]
        r2 = r.copy()
        b2 = b.copy()
        r2[:, level:] = r[:, 1:-level + 1]
        b2[:, level // 2:] = b[:, 1:-level // 2 + 1]
        img[:, :, 0] = r2
        img[:, :, 2] = b2
        img = cv2.GaussianBlur(img, (hsize, hsize), np.sqrt(hsize/6))
        if return_uint8:
            return img.astype(np.uint8)
        else:
            return img


if __name__ == '__main__':
    distor = Distortion()
    distor.generate_data(r"D:\temp_data\precision_data\test\normal", r"D:\temp_data\distortion")