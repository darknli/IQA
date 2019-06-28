# -*- coding:utf-8 -*-
from tensorflow.python import keras
import math
import numpy as np
import cv2
import glob
import os


class DataGenerator(keras.utils.Sequence):
    def __init__(self, origin_dir, distort_dir, batch_size=1, img_shape=(256, 256)):
        self.batch_size = batch_size
        self.filenames = self.get_filenames(origin_dir)
        self.img_shape = img_shape
        self.length = len(self.filenames)
        self.level_dirs = self.get_level_dirs(distort_dir)
        print('has %d examples' % len(self.filenames))

    def get_num_distort_level(self):
        return len(self.level_dirs), len(list(self.level_dirs.items())[0][1])+1

    def get_level_dirs(self, level_dir):
        distor_dirs = glob.glob(os.path.join(level_dir, '*'))
        level_dirs = {}
        for distor_dir in distor_dirs:
            level_dirs[distor_dir] = glob.glob(os.path.join(distor_dir, "*"))
        return level_dirs


    def get_filenames(self, directory):
        return glob.glob(os.path.join(directory, '*'))


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

        origin_img = self.process_image(img)
        img_name = os.path.basename(img)
        for distort, level_paths in self.level_dirs.items():
            level_imgs.append(origin_img)
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


class FTDataGenerator(keras.utils.Sequence):
    def __init__(self, mos_list, distort_dir, batch_size=32, img_shape=(256, 256)):
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.filenames = self.get_files(mos_list, distort_dir)
        self.length = len(self.filenames)

    def get_files(self, mos_list, distort_dir):
        files = [(os.path.join(distort_dir, name),score) for name, score in mos_list]
        return files

    def get_name2score(self, mos_list):
        name2score = {}
        for name, score in mos_list:
            name2score[name] = score
        return name2score

    def __len__(self):
        #计算每一个epoch的迭代次数
        return math.ceil(self.length / float(self.batch_size))

    def __getitem__(self, index):
        # print('get batch size data')
        batch = 0
        data = []
        labels = []
        while batch < self.batch_size:
            try:
                number = np.random.randint(0, self.length)
                img = self.get_iqa_imgs(self.filenames[number][0])
                data.append(img)
                label = self.name2score[self.filenames[number][1]]
                labels.append(np.array([label])/9)
                batch += 1
            except OSError:
                continue
        data = np.array(data)
        labels = np.array(labels)
        return data, labels


    def process_image(self, image):
        """Given an image, process it and return the array."""
        img = cv2.imread(image)
        w, h = img.shape[:2]
        target_w, target_h = self.img_shape
        truncated_w, truncated_h = (np.random.randint(0, w-target_w), np.random.randint(0, h-target_h))
        img = img[truncated_w:truncated_w+target_w, truncated_h:truncated_h+target_h, :]
        img = img.astype(np.float32)/127 - 1
        return img


def get_train_val(filename, ratio=0.8, is_shuffle=True):
    name2score = []
    with open(filename) as f:
        for line in f.read():
            score, name = line.strip().split(' ')
            name2score.append((name, float(score)))
    if is_shuffle:
        np.random.shuffle(name2score)
    length = len(name2score)
    train_size = int(length*ratio)
    return name2score[:train_size], name2score[train_size:]