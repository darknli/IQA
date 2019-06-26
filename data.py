# -*- coding:utf-8 -*-
from tensorflow.python import keras
import math
import numpy as np
import cv2
import glob
import os
from PIL import Image


class DataGenerator(keras.utils.Sequence):
    def __init__(self, directory, classes, batch_size=1, img_shape=(299, 299), is_training=True):
        self.batch_size = batch_size
        self.get_labels(classes)
        self.get_filenames(directory)
        self.img_shape = img_shape
        self.length = len(self.filenames)
        self.num_classes = len(self.label2idx)
        self.is_training = is_training

    def get_labels(self, classes):
        self.label2idx = {}
        for i, label in enumerate(classes):
            self.label2idx[label] = i

    def get_filenames(self, directory):
        self.filenames = []
        self.filelabels = []
        classes = glob.glob(os.path.join(directory, '*'))
        for label_dir in classes:
            label = os.path.split(label_dir)[-1]
            idx = self.label2idx[label]
            files = glob.glob(os.path.join(label_dir, '*'))
            self.filenames += files
            self.filelabels += [idx] * len(files)
            print('%s has %d examples' % (label, len(files)))

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
                number = random.randint(0, self.length-1)
                img = self.process_image(self.filenames[number], self.img_shape)
                data.append(img)
                label = np.zeros((self.num_classes))
                label[self.filelabels[number]] = 1
                labels.append(label)
                batch += 1
            except OSError:
                continue
        return np.array(data), np.array(labels)

    def process_image(self, image, target_shape):
        """Given an image, process it and return the array."""
        cv2.imread(image)
        else:
            img = np.array(img)
        img = img.astype(np.float32)/127 - 1
        return img