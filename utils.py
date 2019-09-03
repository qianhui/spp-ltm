"""Utilities for ADDA."""

import os
import random

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import params
from datasets import get_mnist, get_usps, get_svhn


def denormalize(x, std, mean):
    """Invert normalization, and then convert array into image."""
    out = x * std + mean
    return out.clamp(0, 1)


def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)


def init_model(net, restore):
    """Init models with cuda and weights."""
    # init weights of model
    net.apply(init_weights)

    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()

    return net


def save_model(net, filename):
    """Save trained model."""
    if not os.path.exists(params.model_root):
        os.makedirs(params.model_root)
    torch.save(net.state_dict(),
               os.path.join(params.model_root, filename))
    print("save pretrained model to: {}".format(os.path.join(params.model_root,
                                                             filename)))

def getSourceEncoderPath(encoder, sourceLoader):
    return os.path.join(params.model_root, encoder.getDescription() + "_" + sourceLoader.desc + "_encoder.pt")

def getSourceClassifierPath(encoder, sourceLoader):
    return os.path.join(params.model_root, encoder.getDescription() + "_" + sourceLoader.desc + "_classifier.pt")

def getTargetEncoderPath(encoder, sourceLoader, targetLoader):
    return os.path.join(params.model_root, encoder.getDescription() + "_" + sourceLoader.desc + "_adapt_" + targetLoader.desc + "_encoder.pt")

def getTargetDiscriminatorPath(encoder, sourceLoader, targetLoader):
    return os.path.join(params.model_root, encoder.getDescription() + "_" + sourceLoader.desc + "_adapt_" + targetLoader.desc + "_discriminator.pt")


