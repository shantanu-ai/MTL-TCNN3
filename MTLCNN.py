import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class MTLCNN(nn.Module):
    def __init__(self, init_weights, device):
        super(MTLCNN, self).__init__()

        # conv layers
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.conv1.weight = torch.nn.Parameter(init_weights["conv1_wt"]).to(device)
        self.conv1.bias = torch.nn.Parameter(init_weights["conv1_bias"]).to(device)

        self.bn1 = nn.BatchNorm2d(96)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.conv2.weight = torch.nn.Parameter(init_weights["conv2_wt"]).to(device)
        self.conv2.bias = torch.nn.Parameter(init_weights["conv2_bias"]).to(device)

        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.conv3.weight = torch.nn.Parameter(init_weights["conv3_wt"]).to(device)
        self.conv3.bias = torch.nn.Parameter(init_weights["conv3_bias"]).to(device)

        self.pool = nn.MaxPool2d(3, 2)

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.float().cuda()
        else:
            x = x.float()

        # encoder
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.conv3(x))  # the latent space representation

        return x
