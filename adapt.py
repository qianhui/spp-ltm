"""Adversarial adaptation to train target encoder."""

import os

import torch
import torch.optim as optim
import params
from DigitTransfer.adda.eval import test_model
from DigitTransfer.adda.utils import getTargetEncoderPath, getTargetDiscriminatorPath, getSourceEncoderPath, getSourceClassifierPath
from DigitTransfer.adda.EnDecoder import *
from DigitTransfer.adda.DataSet import *


def train_tgt(src_encoder, tgt_encoder, critic,
              src_data_loader, tgt_data_loader, classifier, tgt_test_data_loader):
    """Train encoder for target domain."""

    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    src_encoder.eval()
    tgt_encoder.train()
    critic.train()

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
                               lr=params.c_learning_rate,
                               betas=(params.beta1, params.beta2))
    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=params.d_learning_rate,
                                  betas=(params.beta1, params.beta2))
    # len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    ####################
    # 2. train network #
    ####################
    #test_model(src_encoder, classifier, tgt_test_data_loader)
    #test_model(tgt_encoder, classifier, tgt_test_data_loader)
    for epoch in range(params.num_epochs):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((images_src, _), (images_tgt, _)) in data_zip:
            ###########################
            # 2.1 train discriminator #
            ###########################

            # make images variable
            images_src = images_src.cuda()
            images_tgt = images_tgt.cuda()
            # print(images_src.shape, images_tgt.shape)

            # zero gradients for optimizer
            optimizer_critic.zero_grad()

            # extract and concat features
            feat_src = src_encoder(images_src)
            feat_tgt = tgt_encoder(images_tgt)
            # print(src_encoder)
            # print(feat_tgt.shape)
            # print(feat_src.shape)
            # print(tgt_encoder)
            # print(feat_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)

            # predict on discriminator
            pred_concat = critic(feat_concat.detach())

            # prepare real and fake label
            label_src = torch.ones(feat_src.size(0)).long()
            label_tgt = torch.zeros(feat_tgt.size(0)).long()
            label_concat = torch.cat((label_src, label_tgt), 0).cuda()

            # compute loss for critic
            loss_critic = criterion(pred_concat, label_concat)
            loss_critic.backward()

            # optimize critic
            optimizer_critic.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            ############################
            # 2.2 train target encoder #
            ############################

            # zero gradients for optimizer
            optimizer_critic.zero_grad()
            optimizer_tgt.zero_grad()

            # extract and target features
            # feat_tgt = tgt_encoder(images_tgt)
            feat_tgt = tgt_encoder(images_tgt)

            # predict on discriminator
            pred_tgt = critic(feat_tgt)

            # prepare fake labels
            label_tgt = torch.ones(feat_tgt.size(0)).long().cuda()

            # compute loss for target encoder
            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()

            # optimize target encoder
            optimizer_tgt.step()

            #######################
            # 2.3 print step info #
            #######################
            if ((step + 1) % params.log_step == 0):
                print("Epoch [{}/{}]:"
                      "d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
                      .format(epoch + 1,
                              params.num_epochs,
                              loss_critic.item(),
                              loss_tgt.item(),
                              acc.item()))
        test_model(tgt_encoder, classifier, tgt_test_data_loader)
        #############################
        # 2.4 save model parameters #
        #############################
        """
        if ((epoch + 1) % params.save_step == 0):
            torch.save(critic.state_dict(), os.path.join(
                params.model_root,
                "ADDA-critic-{}.pt".format(epoch + 1)))
            torch.save(tgt_encoder.state_dict(), os.path.join(
                params.model_root,
                "ADDA-target-encoder-{}.pt".format(epoch + 1)))
        """
    tgt_encoder_path = getTargetEncoderPath(src_encoder, src_data_loader, tgt_data_loader)
    tgt_critic_path = getTargetDiscriminatorPath(src_encoder, src_data_loader, tgt_data_loader)
    torch.save(critic.state_dict(), tgt_critic_path)
    torch.save(tgt_encoder.state_dict(), tgt_encoder_path)
    return tgt_encoder


def adapt_train(src_encoder, tgt_encoder, src_data_loader, tgt_data_loader, tgt_test_data_loader):

    classifier = Classifier().to(args.device)

    src_encoder_path = getSourceEncoderPath(src_encoder, src_data_loader)
    src_classifier_path = getSourceClassifierPath(src_encoder, src_data_loader)

    checkpoint = torch.load(src_encoder_path)
    src_encoder.load_state_dict(checkpoint['net'])
    src_encoder.fc1.cx = checkpoint['cx']
    tgt_encoder.load_state_dict(checkpoint['net'])
    tgt_encoder.fc1.cx = checkpoint['cx']

    checkpoint = torch.load(src_classifier_path)
    classifier.load_state_dict(checkpoint)
    # print(src_encoder.fc1.cx)

    critic = Discriminator(args.d_input_dims, args.d_hidden_dims, args.d_output_dims).to(args.device)
    train_tgt(src_encoder, tgt_encoder, critic,
              src_data_loader, tgt_data_loader, classifier, tgt_test_data_loader)
#
if __name__ == '__main__':
    src_data_loader = getMnistTrain()
    tgt_data_loader = getuspsTrain()
    tgt_test_data_loader = getuspsTest()
    src_encoder = SPPLTMEncoder().to(args.device)
    tgt_encoder = SPPLTMEncoder().to(args.device)
    adapt_train(src_encoder, tgt_encoder, src_data_loader, tgt_data_loader, tgt_test_data_loader)


    src_data_loader = getuspsTrain()
    tgt_data_loader = getMnistTrain()
    tgt_test_data_loader = getMnistTest()
    src_encoder = SPPLTMEncoder().to(args.device)
    tgt_encoder = SPPLTMEncoder().to(args.device)
    adapt_train(src_encoder, tgt_encoder, src_data_loader, tgt_data_loader, tgt_test_data_loader)

