import os.path
from os import listdir
from os.path import join
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from skimage.transform import resize
from utils import add_noise, add_sampling, add_blur

# 확장자 판별 return: bool
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

class Dataset(Dataset):
    def __init__(self, data_dir, transform=None, task=None, opts=None):
        super(Dataset, self).__init__()
        self.data_dir = data_dir
        self.lst_data = [x for x in os.listdir(data_dir)]
        self.transform = transform
        self.totensor = ToTensor()
        self.task = task
        self.opts = opts

    def __len__(self):
        return len(self.lst_data)

    def __getitem__(self, index):

        img = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))[:, :, :3]

        if img.dtype == np.uint8:
            img = img / 255.0

        if img.ndim == 2:
            img = img[:, :, np.newaxis]

        label = img

        data = {'label': label}

        if self.task == 'denoising':
            data['input'] = add_noise(data['label'], type=self.opts[0], opts=self.opts[1])
        elif self.task == 'inpainting':
            data['input'] = add_sampling(data['label'], type=self.opts[0], opts=self.opts[1])

        if self.transform:
            img = self.transform(img)

        if self.task == 'super_resolution':
            data['input'] = add_blur(data['label'], type=self.opts[0], opts=self.opts[1])

        img = self.totensor(img)
        return img


class ToTensor(object):
    def __call__(self, data):
        for key, value in data.items():
            value = value.transpose((2, 0, 1)).astype(np.float32)
            data[key] = torch.from_numpy(value)

        return data


class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        for key, value in data.items():
            data[key] = (value - self.mean) / self.std

        return data


class RandomCrop(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        h, w = data['label'].shape[:2]
        new_h, new_w = self.shape

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis]
        id_x = np.arange(left, left + new_w, 1)

        for key, value in data.items():
            data[key] = value[id_y, id_x]

        return data


class Resize(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        for key, value in data.items():
            data[key] = resize(value, output_shape=(self.shape[0], self.shape[1], self.shape[2]))

        return data