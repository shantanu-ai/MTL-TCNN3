import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class MultitaskClassifier(nn.Module):
    def __init__(self, labels):
        super(MultitaskClassifier, self).__init__()

        texture_labels = labels["texture_labels"]
        image_net_labels = labels["image_net_labels"]

        # shared conv layers
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        # self.conv1.weight = torch.nn.Parameter(init_weights["conv1_wt"]).to(device)
        # self.conv1.bias = torch.nn.Parameter(init_weights["conv1_bias"]).to(device)

        self.bn1 = nn.BatchNorm2d(96)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        # self.conv2.weight = torch.nn.Parameter(init_weights["conv2_wt"]).to(device)
        # self.conv2.bias = torch.nn.Parameter(init_weights["conv2_bias"]).to(device)

        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        # self.conv3.weight = torch.nn.Parameter(init_weights["conv3_wt"]).to(device)
        # self.conv3.bias = torch.nn.Parameter(init_weights["conv3_bias"]).to(device)

        self.max_pool = nn.MaxPool2d(3, 2)
        self.avg_pool = nn.AvgPool2d(13, 1)

        self.drop_out = nn.Dropout(p=0.5)

        # Object detection task
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.object_detect_fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.object_detect_fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.object_detect_out = nn.Linear(in_features=4096, out_features=len(image_net_labels))

        # Texture classification task
        self.fc_texture_1 = nn.Linear(in_features=384, out_features=4096)
        self.fc_texture_2 = nn.Linear(in_features=4096, out_features=4096)
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
        x = F.relu(self.conv3(x))

        # Object detection task
        object_head = F.relu(self.conv4(x))

        object_head = F.relu(self.conv5(object_head))
        object_head = self.max_pool(object_head)
        object_head = self.drop_out(object_head)

        object_head = self.object_detect_fc1(object_head.reshape(-1, 256 * 6 * 6))
        object_head = self.drop_out(object_head)
        object_head = F.relu(object_head)
        object_head = F.relu(self.object_detect_fc2(object_head))
        object_head = F.relu(self.object_detect_out(object_head))

        # texture classification
        texture_head = self.avg_pool(x)
        texture_head = texture_head.reshape(-1, 384 * 1)
        texture_head = F.relu(self.fc_texture_1(texture_head))
        texture_head = F.relu(self.fc_texture_2(texture_head))
        texture_head = self.texture_out(texture_head)

        return object_head, texture_head
