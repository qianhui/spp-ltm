# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn.functional as F
from EnDecoder import *
from DataSet import *
from DataSet import getAllTest
import os
import params as args
from utils import getTargetEncoderPath, getTargetDiscriminatorPath, getSourceEncoderPath, getSourceClassifierPath

def test_model(encoder, classifier, test_loader):
    encoder.to(args.device)
    classifier.to(args.device)
    encoder.eval()
    classifier.eval()
    test_loss = 0
    correct = 0
    # ten = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            if "Channel" in encoder.getDescription():
                code, c1loss, c2loss = encoder(data)
                output = classifier(code)
            else:
                output = classifier(encoder(data))
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # if pred is 10:
                # ten += ten
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('Test set: ' + test_loader.desc + ' Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    dataLoader = getMnistTrain()

    encoder = Encoder().to(args.device)
    classifier = Classifier().to(args.device)

    encoder_path = getSourceEncoderPath(encoder, dataLoader)
    classifier_path = getSourceClassifierPath(encoder, dataLoader)

    encoder.load_state_dict(torch.load(encoder_path))
    classifier.load_state_dict(torch.load(classifier_path))

    for generalSet in getAllTest():
        test_model(encoder, classifier, generalSet)

