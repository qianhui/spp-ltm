# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn.functional as F
from DigitTransfer.adda.EnDecoder import *
from DigitTransfer.adda.DataSet import *
import os
import DigitTransfer.adda.params as args
from DigitTransfer.adda.eval import test_model
from DigitTransfer.adda.utils import *
from torch.optim import lr_scheduler

def train_src(encoder, classifier, train_loader, optimizer, epoch):
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad()
        if "Channel" in encoder.getDescription():
            code, c1loss, c2loss = encoder(data)
            output = classifier(code)
            realLoss = F.nll_loss(output, target)
            loss = realLoss - 10 * c1loss - 10 * c2loss
            # print(loss, c1loss, c2loss)
            losses.append(realLoss.item())
        else:
            output = classifier(encoder(data))
            loss = F.nll_loss(output, target)
            losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    # print(losses)


def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)

def modelTrain(encoder, classifier, dataSet):
    # dataSet = getSVHN()
    trainSet, testSet = dataSet
    torch.manual_seed(args.manual_seed)
    # encoder.apply(init_weights)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()),
                                 lr=args.c_learning_rate, betas=(args.beta1, args.beta2))
    # optimizer = torch.optim.SGD(list(encoder.parameters()) + list(classifier.parameters()), lr=args.c_learning_rate, momentum=0.5)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    encoder.to(args.device)
    classifier.to(args.device)
    encoder.train()
    classifier.train()
    for epoch in range(1, args.num_epochs_pre + 1):
        train_src(encoder, classifier, trainSet, optimizer, epoch)
        test_model(encoder, classifier, trainSet)
        # test_model(encoder, classifier, testSet)
        scheduler.step()
        for generalSet in getAllTest():
            test_model(encoder, classifier, generalSet)
        # print(encoder.fc1.cx)
        state = {
            'net': encoder.state_dict(),
            'cx': encoder.fc1.cx
        }
        torch.save(state, getSourceEncoderPath(encoder, trainSet))
        torch.save(classifier.state_dict(), getSourceClassifierPath(encoder, trainSet))
        # print(encoder.fc1.cx)

if __name__ == '__main__':

    dataSet = getMNIST()
    encoder = SPPLTMEncoder()
    classifier = Classifier()
    modelTrain(encoder, classifier, dataSet)

    dataSet = getUSPS()
    encoder = SPPLTMEncoder()
    classifier = Classifier()
    modelTrain(encoder, classifier, dataSet)


