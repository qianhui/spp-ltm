# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import Parameter
from math import floor, ceil
from torch.nn import init
from torch import Tensor
import math


class Encoder(nn.Module):
    def __init__(self, classes=10):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.bn1 = nn.BatchNorm2d(20)
        self.bn2 = nn.BatchNorm2d(50)
        self.fc1 = nn.Linear(5 * 5 * 50, 500)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.dropout2d(x)
        x = F.max_pool2d(x, 2, 2)
        # x = x.view(-1, 5 * 5 * 50)
        x = x.view(x.size()[0], -1)
        # x = x.view(4 * 4 * 50, -1)
        x = self.fc1(x)
        return x

    def getDescription(self):
        return "LeEncoder"

class Classifier(nn.Module):
    """LeNet classifier model for ADDA."""

    def __init__(self, hidden_size=500):
        """Init LeNet encoder."""
        super(Classifier, self).__init__()
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = F.dropout(F.relu(x), training=self.training)
        # x = F.relu(x)
        out = self.fc2(x)
        return F.log_softmax(out, dim=1)

    def getDescription(self):
        return "Classifier"

class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims, hidden_dims, output_dims=2):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            # nn.Linear(hidden_dims, hidden_dims),
            # nn.ReLU(),
            nn.Linear(hidden_dims, output_dims),
            nn.LogSoftmax()
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out

class LTM(nn.Module):
    """Naive LSTM like nn.LSTM"""

    def __init__(self, input_size: int, hidden_size: int):
        super(LTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_gate = nn.Linear(input_size, hidden_size)
        self.forget_gate = nn.Linear(input_size, hidden_size)
        self.output_gate = nn.Linear(input_size, hidden_size)
        self.self_gate = nn.Linear(input_size, hidden_size)

        self.cx = torch.zeros(1, hidden_size).cuda()
        # self.cx.requires_grad = False
        self.traincx = True

        self.reset_weigths()

    def reset_weigths(self):
        """reset weights
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, inputs):
        """Forward
        Args:
            inputs: [1, 1, input_size]
            state: ([1, 1, hidden_size], [1, 1, hidden_size])
        """
        #         seq_size, batch_size, _ = inputs.size()
        #    A @ B = A x B, A * B = A pointwise B
        x = inputs
        self.cx = self.cx.detach()
        # input gate
        i = torch.sigmoid(self.input_gate(x))
        # forget gate
        f = torch.sigmoid(self.forget_gate(x))
        # cell
        g = torch.tanh(self.self_gate(x))
        # output gate
        o = torch.sigmoid(self.output_gate(x))
        c_next = f * self.cx + i * g
        h_next = o * torch.tanh(c_next)
        if self.traincx and self.training:
            self.cx = torch.mean(c_next, dim=0)
        return h_next

class LTMEncoder(nn.Module):
    def __init__(self, classes=10):
        super(LTMEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 3, 1)
        self.conv2 = nn.Conv2d(20, 50, 3, 1)
        self.bn1 = nn.BatchNorm2d(20)
        self.bn2 = nn.BatchNorm2d(50)
        self.fc1 = LTM(6 * 6 * 50, 500)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.dropout2d(x)
        x = F.max_pool2d(x, 2, 2)
        # x = x.view(-1, 5 * 5 * 50)
        x = x.view(x.size()[0], -1)
        # x = x.view(4 * 4 * 50, -1)
        x = self.fc1(x)
        return x

    def getDescription(self):
        return "LTMEncoder"

class SpatialPyramidPooling2d(nn.Module):

    def __init__(self, num_level, pool_type='max_pool'):
        super(SpatialPyramidPooling2d, self).__init__()
        self.num_level = num_level
        self.pool_type = pool_type

    def forward(self, x):
        N, C, H, W = x.size()
        for i in range(self.num_level):
            level = i + 1
            kernel_size = (ceil(H / level), ceil(W / level))
            stride = (ceil(H / level), ceil(W / level))
            padding = (floor((kernel_size[0] * level - H + 1) / 2), floor((kernel_size[1] * level - W + 1) / 2))
            # print(level, kernel_size, stride, padding)
            if self.pool_type == 'max_pool':
                tensor = (F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)).view(N, -1)
            else:
                tensor = (F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)).view(N, -1)
            # print(tensor.shape)
            if i == 0:
                res = tensor
            else:
                res = torch.cat((res, tensor), 1)
        return res

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'num_level = ' + str(self.num_level) \
               + ', pool_type = ' + str(self.pool_type) + ')'

class SPPLTMEncoder(nn.Module):
    def __init__(self, num_level=3, pool_type='max_pool', classes=10):
        super(SPPLTMEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 3, 1)
        self.conv2 = nn.Conv2d(20, 50, 3, 1)
        self.bn1 = nn.BatchNorm2d(20)
        self.bn2 = nn.BatchNorm2d(50)
        self.num_grid = self._cal_num_grids(num_level)
        self.spp_layer = SpatialPyramidPooling2d(num_level)
        self.fc1 = LTM(700, 500)

    def _cal_num_grids(self, level):
        count = 0
        for i in range(level):
            count += (i + 1) * (i + 1)
        return count

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.max_pool2d(x, 2, 2)
        # x = x.view(-1, 5 * 5 * 50)
        x = self.spp_layer(x)
        # x = x.view(x.size()[0], -1)
        # print(x.shape, self.num_grid)
        # x = x.view(4 * 4 * 50, -1)
        x = self.fc1(x)
        return x

    def getDescription(self):
        return "SPPLTMEncoder"

class SPPEncoder(nn.Module):
    def __init__(self, num_level=3, pool_type='max_pool', classes=10):
        super(SPPEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 3, 1)
        self.conv2 = nn.Conv2d(20, 50, 3, 1)
        self.bn1 = nn.BatchNorm2d(20)
        self.bn2 = nn.BatchNorm2d(50)
        self.num_grid = self._cal_num_grids(num_level)
        self.spp_layer = SpatialPyramidPooling2d(num_level)
        self.fc1 = nn.Linear(700, 500)

    def _cal_num_grids(self, level):
        count = 0
        for i in range(level):
            count += (i + 1) * (i + 1)
        return count

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.max_pool2d(x, 2, 2)
        # x = x.view(-1, 5 * 5 * 50)
        x = self.spp_layer(x)
        # x = x.view(x.size()[0], -1)
        # print(x.shape, self.num_grid)
        # x = x.view(4 * 4 * 50, -1)
        x = self.fc1(x)
        return x

    def getDescription(self):
        return "SPPEncoder"

class Baseline(nn.Module):
    """Naive LSTM like nn.LSTM"""

    def __init__(self, input_size: int, hidden_size: int):
        super(Baseline, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_gate = nn.Linear(input_size, hidden_size)
        self.self_gate = nn.Linear(input_size, hidden_size)

        self.cx = torch.randn(1, hidden_size).cuda()

        self.reset_weigths()

    def reset_weigths(self):
        """reset weights
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, inputs):
        """Forward
        Args:
            inputs: [1, 1, input_size]
            state: ([1, 1, hidden_size], [1, 1, hidden_size])
        """
        #         seq_size, batch_size, _ = inputs.size()
        #    A @ B = A x B, A * B = A pointwise B
        x = inputs
        # cell
        # g = torch.tanh(self.self_gate(x))
        # output gate
        o = torch.sigmoid(self.output_gate(x))
        h_next = o * torch.tanh(self.cx)
        return h_next

class SPPBaseEncoder(nn.Module):
    def __init__(self, num_level=3, pool_type='max_pool', classes=10):
        super(SPPBaseEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 3, 1)
        self.conv2 = nn.Conv2d(20, 50, 3, 1)
        self.bn1 = nn.BatchNorm2d(20)
        self.bn2 = nn.BatchNorm2d(50)
        self.num_grid = self._cal_num_grids(num_level)
        self.spp_layer = SpatialPyramidPooling2d(num_level)
        self.fc1 = Baseline(700, 500)

    def _cal_num_grids(self, level):
        count = 0
        for i in range(level):
            count += (i + 1) * (i + 1)
        return count

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.max_pool2d(x, 2, 2)
        # x = x.view(-1, 5 * 5 * 50)
        x = self.spp_layer(x)
        # x = x.view(x.size()[0], -1)
        # print(x.shape, self.num_grid)
        # x = x.view(4 * 4 * 50, -1)
        x = self.fc1(x)
        return x

    def getDescription(self):
        return "SPPBaseEncoder"


