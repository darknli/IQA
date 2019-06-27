# -*- coding:utf-8 -*-
from tensorflow.python import keras
import math
import numpy as np
import cv2
import glob
import os
from PIL import Image


class DataGenerator(keras.utils.Sequence):
    def __init__(self, origin_dir, distort_dir, batch_size=1, img_shape=(299, 299)):
        self.batch_size = batch_size
        self.get_filenames(origin_dir)
        self.img_shape = img_shape
        self.length = len(self.filenames)
        self.level_dirs = self.get_level_dirs(distort_dir)

    def get_num_distort_level(self):
        return len(self.level_dirs), len(list(self.level_dirs.items())[0][0])

    def get_level_dirs(self, level_dir):
        distor_dirs = glob.glob(os.path.join(level_dir, '*'))
        level_dirs = {}
        for distor_dir in distor_dirs:
            level_dirs[distor_dir] = glob.glob(os.path.join(distor_dir, "*"))
        return level_dirs


    def get_filenames(self, directory):
        self.filenames = glob.glob(os.path.join(directory, '*'))
        print('has %d examples' % len(self.filenames))


    def __len__(self):
        #计算每一个epoch的迭代次数
        return math.ceil(self.length / float(self.batch_size))

    def __getitem__(self, index):
        # print('get batch size data')
        batch = 0
        data = []
        while batch < self.batch_size:
            try:
                number = np.random.randint(0, self.length)
                img = self.get_iqa_imgs(self.filenames[number])
                data.append(img)
                batch += 1
            except OSError:
                continue
        data = np.concatenate(data, axis=0)
        label = np.zeros((len(data)))
        return data, label

    def get_iqa_imgs(self, img):
        level_imgs = []

        level_imgs.append(self.process_image(img))
        img_name = os.path.basename(img)
        for distort, level_paths in self.level_dirs.items():
            for level_path in level_paths:
                level_img = os.path.join(level_path, img_name)
                level_imgs.append(self.process_image(level_img))

        return np.array(level_imgs)



    def process_image(self, image):
        """Given an image, process it and return the array."""
        img = cv2.imread(image)
        w, h = img.shape[:2]
        target_w, target_h = self.img_shape
        truncated_w, truncated_h = (np.random.randint(0, w-target_w), np.random.randint(0, h-target_h))
        img = img[truncated_w:truncated_w+target_w, truncated_h:truncated_h+target_h, :]
        img = img.astype(np.float32)/127 - 1
        return img