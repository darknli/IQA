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
    mblur_level = [];  % # 24 Motion Blur
"""
import numpy as np
import cv2
import os
from PIL import Image
from glob import glob
from tqdm import tqdm
from time import sleep
from multiprocessing import Process

class Distortion:
    def __init__(self):
        """
        1:2
        3:3
        4:0
        7:3

        """
        self.wn_level = [0.03, 0.06, 0.1, 0.2]
        self.gnc_level = [0.0140, 0.0198, 0.0343, 0.0524]
        self.hfn_level = [0.03, 0.06, 0.1, 0.2]
        self.in_level = [0.001, 0.01, 0.05, 0.1]
        self.qn_level = [15, 25, 35, 45]
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
        self.mblur = [5, 8, 12, 16]
        self.idx2func = {
            1: "gaussian_noise",
            2: "gaussian_blur",
            3: "impulse_noise",
            4: "jpeg_compression",
            5: "nepn",
            6: "block_wise",
            7: "hf_noise",
            8: "multi_gn",
            9:"cqd",
            10:"ca",
            11:"quantization_noise",
            12:"motion_blur"
        }

    def generate_data(self, data, rank_root, num_workers=4, sorts=[]):
        imgs = glob(os.path.join(data, '*'))
        if not os.path.exists(rank_root):
            os.mkdir(rank_root)

        workers = []
        end = 0
        img_length = len(imgs)
        workload = img_length//num_workers
        for i in range(num_workers-1):
            begin = end
            end += workload
            workers.append(
                Process(target=self.generate_data_one_worker, args=(imgs[begin:end], rank_root, i, sorts))
            )
        workers.append(
            Process(target=self.generate_data_one_worker, args=(imgs[end:img_length], rank_root, num_workers-1, sorts))
        )
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

    def generate_data_one_worker(self, imgs, rank_root, worker_No=-1, sorts=[]):
        print('Process No.%d start...' % worker_No)
        if not os.path.exists(rank_root):
            os.mkdir(rank_root)

        if len(sorts) == 0:
            sorts = [i for i in range(1, len(self.idx2func))]
        for i in sorts:
            func_path = os.path.join(rank_root, str(i))
            if not os.path.exists(func_path):
                try:
                    os.mkdir(func_path)
                except FileExistsError:
                    pass
            # print('generating the %s ...' % self.idx2func[i])
            # with tqdm(total=process_length) as pbar:
            for level in range(4):
                level_path = os.path.join(func_path, str(level))
                if not os.path.exists(level_path):
                    try:
                        os.mkdir(level_path)

                    except FileExistsError:
                        pass
                for path in imgs:
                    if 'gif' in path:
                        continue
                    img_name = os.path.basename(path)
                    try:
                        img = cv2.imread(path).astype(np.float)
                    except AttributeError:
                        print('%s 失败'%path)
                        os.remove(path)
                        continue
                        # raise AttributeError('%s 无法读取'%img)
                    distorted_img = eval("self."+self.idx2func[i])(img, level, worker_No)
                    save_path = os.path.join(level_path, img_name)
                    if not os.path.exists(save_path):
                        cv2.imwrite(save_path, distorted_img)
                        # pbar.update(1)
        print('Process No.%d done' % worker_No)

    def gaussian_noise(self, img, level, worker_No=-1, is_rgb=True, return_uint8=True, var=False):
        """
        高斯噪声
        :param img: 输入图像rgb矩阵
        :param level: 方差等级
        :param is_rgb: 灰度或彩色
        """
        img = img.astype(np.float)/255.0
        shape = img.shape
        if not var:
            level = self.wn_level[level]
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

    def gaussian_blur(self, img, level, worker_No=-1, return_uint8=True):
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

    def impulse_noise(self, img, level, worker_No=-1, return_uint8=True):
        """
        脉冲噪声（椒盐噪声），随机把像素点变成0或255
        :param img: 输入图像rgb矩阵
        :param level: 噪声比例等级
        """
        salt_prob = np.random.uniform(0, self.in_level[level])
        salt_noise = np.zeros_like(img)
        pepper_noise = 255 * np.ones_like(img)
        prob_mat = np.random.uniform(0, 1, img.shape[:2]+(1,))
        pepper_img = np.where(prob_mat<self.in_level[level], pepper_noise, img)
        img = np.where(prob_mat<salt_prob, salt_noise, pepper_img)
        if return_uint8:
            return img.astype(np.uint8)
        else:
            return img

    def quantization_noise(self, img, level, worker_No=-1, return_uint8=True):
        """
        量化噪声，把介于某个区间的像素点去上限
        :param img: 输入图像rgb矩阵
        :param level: 噪声比例等级
        """
        img = img.astype(np.float)
        img = np.floor(img/self.qn_level[level])
        img *= self.qn_level[level]
        if return_uint8 == True:
            return img.astype(np.uint8)
        else:
            return img

    def jpeg_compression(self, img, level, worker_No=-1, return_uint8=True):
        """
        jpeg压缩
        :param img: 输入图像rgb矩阵
        :param level: 压缩等级
        """
        cv2.imwrite("jpeg_temp_%d.jpg"%worker_No, img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_level[level]])
        img = cv2.imread("jpeg_temp_%d.jpg"%worker_No)
        os.remove("jpeg_temp_%d.jpg"%worker_No)
        if return_uint8:
            return img.astype(np.uint8)
        else:
            return img

    def nepn(self, img, level, worker_No=-1, return_uint8=True):
        """
        效果：在图像上生成多个“像其他图像遗留在这里”的图，效果不错
        """
        w, h = img.shape[:2]
        nepn_img = img.copy()
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

    def block_wise(self, img, level, worker_No=-1, return_uint8=True):
        """
        效果：在图片上中生成多个随机的小方块，看起来很突兀，可以不要
        """
        w, h = img.shape[:2]

        r_mean = np.mean(img[:, :, 0])
        g_mean = np.mean(img[:, :, 1])
        b_mean = np.mean(img[:, :, 2])

        level_map = {
            4: lambda x: x,
            3: lambda x: x,
            2: lambda x: x + 30,
            1: lambda x: x + 50,
            0: lambda x: np.zeros_like(x)
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

    def hf_noise(self, img, level, worker_No=-1, return_uint8=True):
        """
        高频噪声
        """
        # def ghp(img, thresh):
        #
        #     res = d * img
        #     return res

        img = img.astype(np.float) / 255
        img_fft = np.fft.fft2(img)
        thresh = 100
        r, c = img.shape[:2]
        d0 = thresh
        rm = np.tile(np.arange(1, r + 1).reshape(-1, 1), (1, c)) - r / 2 * np.ones((r, c))
        cm = np.tile(np.arange(1, c + 1).reshape(1, -1), (r, 1)) - r / 2 * np.ones((r, c))
        d = np.sqrt(rm ** 2 + cm ** 2)
        divisor = 2 * (d0 ** 2)
        d = 1 - np.exp(-d ** 2 / divisor)
        print(d.shape)
        ghp = np.expand_dims(d, axis=-1)*img_fft
        ifft2 = np.fft.ifft2(ghp)
        img = np.real(ifft2)
        img = np.clip(255 * img, 0, 255)
        img = self.gaussian_noise(img, self.hfn_level[level], worker_No, var=True)
        if return_uint8:
            return img.astype(np.uint8)
        else:
            return img

    def img_denoising(self, img, level, worker_No=-1,  return_uint8=True):
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

    def multi_gn(self, img, level, worker_No=-1, return_uint8=True):
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

    def cqd(self, img, level, worker_No, return_uint8=True):
        """
        Color quantization dither
        """
        cv2.imwrite("cqd_%d.png"%worker_No, img)
        lena = Image.open("cqd_%d.png"%worker_No)
        lena_P_dither = lena.convert("P", palette=Image.ADAPTIVE, colors=self.cqd_level[level])
        img = lena_P_dither.convert("RGB")
        img.save("cqd_%d_1.png"%worker_No)
        img = cv2.imread("cqd_%d_1.png"%worker_No)
        del lena, lena_P_dither
        os.remove("cqd_%d.png"%worker_No)
        os.remove("cqd_%d_1.png"%worker_No)
        if return_uint8:
            return img
        else:
            return img.astype(np.float)

    def ca(self, img, level, worker_No=-1, return_uint8=True):
        hsize = 3
        r = img[:, :, 0]
        b = img[:, :, 2]
        r2 = r.copy()
        b2 = b.copy()
        r2[:, self.ca_level[level]:] = r[:, :-self.ca_level[level]]
        b2[:, self.ca_level[level] // 2:] = b[:, :-self.ca_level[level] // 2]
        img[:, :, 0] = r2
        img[:, :, 2] = b2
        img = cv2.GaussianBlur(img, (hsize, hsize), np.sqrt(hsize/6))
        if return_uint8:
            return img.astype(np.uint8)
        else:
            return img

    def motion_blur(self, img, level, worker_No=-1, return_uint8=True, angle=45):
        image = np.array(img)

        # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
        M = cv2.getRotationMatrix2D((self.mblur[level] / 2, self.mblur[level] / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(self.mblur[level]))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (self.mblur[level], self.mblur[level]))

        motion_blur_kernel = motion_blur_kernel / self.mblur[level]
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)

        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        if return_uint8:
            img = blurred.astype(np.uint8)
        else:
            img = img
        return img


def copy(src, dest):
    import shutil
    files = glob(os.path.join(src, '*'))
    np.random.shuffle(files)
    for file in files[:10000]:
        filename = os.path.basename(file)
        dst_file = os.path.join(dest, filename)
        shutil.copy(file, dst_file)
        print('%s copy to %s'%(file, dst_file))

def check_imgs(dir):
    files = glob(os.path.join(dir, '*'))
    for file in files:
        if cv2.imread(file) == None:
            print('% error'%file)
            os.remove(file)


if __name__ == '__main__':
    distor = Distortion()

    from time import time
    t1 = time()
    distor.generate_data(r"D:\temp_data\iqa\val\origin", r"D:\temp_data\iqa\val\distortion", sorts=[11, 12], num_workers=8)
    t2 = time()
    print(t2-t1)
    # distor.generate_data_deprecated(r"D:\AAA\Data\myiqa\val\origin", r"D:\AAA\Data\myiqa\val\distortion")
    # t3 = time()
    # print(t3-t2)