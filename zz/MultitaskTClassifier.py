import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class MultitaskTClassifier(nn.Module):
    def __init__(self, labels):
        super(MultitaskTClassifier, self).__init__()

        texture_labels = labels["texture_labels"]
        image_net_labels_S2 = labels["image_net_labels_S2"]
        image_net_labels_T = labels["image_net_labels_T"]

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

        # Imagenet S2 detection task
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.object_detect_fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.object_detect_fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.imagenetS2 = nn.Linear(in_features=4096, out_features=len(image_net_labels_S2))

        # Imagenet T detection task
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.object_detect_fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.object_detect_fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.imagenetT = nn.Linear(in_features=4096, out_features=len(image_net_labels_T))

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

        # Imagenet S2 detection task
        imagenetS2_head = F.relu(self.conv4(x))

        imagenetS2_head = F.relu(self.conv5(imagenetS2_head))
        imagenetS2_head = self.max_pool(imagenetS2_head)
        imagenetS2_head = self.drop_out(imagenetS2_head)

        imagenetS2_head = self.object_detect_fc1(imagenetS2_head.reshape(-1, 256 * 6 * 6))
        imagenetS2_head = self.drop_out(imagenetS2_head)
        imagenetS2_head = F.relu(imagenetS2_head)
        imagenetS2_head = F.relu(self.object_detect_fc2(imagenetS2_head))
        imagenetS2_head = F.relu(self.imagenetS2(imagenetS2_head))

        # Imagenet T detection task
        imagenetT_head = F.relu(self.conv4(x))

        imagenetT_head = F.relu(self.conv5(imagenetT_head))
        imagenetT_head = self.max_pool(imagenetT_head)
        imagenetT_head = self.drop_out(imagenetT_head)

        imagenetT_head = self.object_detect_fc1(imagenetT_head.reshape(-1, 256 * 6 * 6))
        imagenetT_head = self.drop_out(imagenetT_head)
        imagenetT_head = F.relu(imagenetT_head)
        imagenetT_head = F.relu(self.object_detect_fc2(imagenetT_head))
        imagenetT_head = F.relu(self.imagenetT(imagenetT_head))

        # texture classification
        texture_head = self.avg_pool(x)
        texture_head = texture_head.reshape(-1, 384 * 1)
        texture_head = F.relu(self.fc_texture_1(texture_head))
        texture_head = F.relu(self.fc_texture_2(texture_head))
        texture_head = self.texture_out(texture_head)

        return imagenetS2_head, imagenetT_head, texture_head
