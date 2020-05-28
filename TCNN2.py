import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class TCNN2(nn.Module):
    def __init__(self, texture_labels):
        super(TCNN2, self).__init__()

        # shared conv layers
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        # torch.nn.init.xavier_uniform(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(96)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        # torch.nn.init.xavier_uniform(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(256)

        self.max_pool = nn.MaxPool2d(3, 2)
        self.drop_out = nn.Dropout(p=0.5)

        self.avg_pool = nn.AvgPool2d(13, 1)

        # Texture classification task
        self.fc_texture_1 = nn.Linear(in_features=256, out_features=4096)
        # torch.nn.init.xavier_uniform(self.fc_texture_1.weight)
        self.fc_texture_2 = nn.Linear(in_features=4096, out_features=4096)
        # torch.nn.init.xavier_uniform(self.fc_texture_2.weight)
        self.texture_out = nn.Linear(in_features=4096, out_features=len(texture_labels))

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.float().cuda()
        else:
            x = x.float()

        # shared conv
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.max_pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.max_pool(x)

        # texture classification

        texture_head = self.avg_pool(x)
        texture_head = texture_head.reshape(-1, 256 * 1)

        texture_head = F.relu(self.fc_texture_1(texture_head))
        texture_head = F.relu(self.fc_texture_2(texture_head))
        texture_head = self.texture_out(texture_head)

        return texture_head
