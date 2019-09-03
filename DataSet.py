# -*- coding: utf-8 -*-
import torchvision.datasets as datasets
from torchvision import transforms
import numpy as np
from CustomDataSets import USPS, CustomCIFAR10
import random
import params as args
import torch

class RandomReverse(object):
    def __init__(self, p=0.3):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            return img * -1
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


mnistTransform = transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))])
uspsTransform = transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.2539,), (0.3842,))])

# trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=None)
mainPath = "data/"

kwargs = {'num_workers': 1, 'pin_memory': 1} if args.cuda else {}

# 0.1307, 0.3081
def getMnistTrain():
    mnistTrain = datasets.MNIST(root=mainPath + 'mnist', train=True, download=False, transform=mnistTransform)
    train_loader = torch.utils.data.DataLoader(
        mnistTrain,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    train_loader.desc = "mnist"
    return train_loader
def getMnistTest():
    mnistTest = datasets.MNIST(root=mainPath + 'mnist', train=False, download=False, transform=mnistTransform)
    train_loader = torch.utils.data.DataLoader(
        mnistTest,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    train_loader.desc = "mnist"
    return train_loader
def getMNIST():
    return getMnistTrain(), getMnistTest()


# 0.2539, 0.3842
def getuspsTrain():
    uspsTrain = USPS(root=mainPath, train=True, transform=uspsTransform)
    train_loader = torch.utils.data.DataLoader(
        uspsTrain,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    train_loader.desc = "usps"
    return train_loader
def getuspsTest():
    uspsTest = USPS(root=mainPath, train=False, transform=uspsTransform)
    train_loader = torch.utils.data.DataLoader(
        uspsTest,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    train_loader.desc = "usps"
    return train_loader
def getUSPS():
    return getuspsTrain(), getuspsTest()

# data.Subset
def getAllTest(): #
    return [getMnistTest(), getuspsTest()]

