# -*- coding: utf-8 -*-
import numpy as np
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import torchvision.datasets as datasets
import os
import os.path
import sys


class USPS(VisionDataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(USPS, self).__init__(root)
        self.filename = "usps.h5"
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        import h5py
        with h5py.File(root + self.filename, 'r') as hf:
            if self.train:
                train = hf.get('train')
                self.data = train.get('data')[:]
                self.labels = train.get('target')[:]
                # print(self.data.shape)
            else:
                test = hf.get('test')
                self.data = test.get('data')[:]
                self.labels = test.get('target')[:]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.labels[index])
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = img.reshape([16, 16])
        # print(img * 255)
        # must [0, 255], uint8
        img = Image.fromarray(np.uint8(img * 255), mode='L')
        # print(np.asarray(img))
        # img = img.reshape([16, 16])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)




