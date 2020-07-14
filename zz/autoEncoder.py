import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # encoder layers
        self.enc1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.bn1 = nn.BatchNorm2d(96)
        self.enc2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.enc3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(3, 2)

        # decoder layers
        self.dec1 = nn.ConvTranspose2d(384, 256, kernel_size=3, stride=2)
        self.dec2 = nn.ConvTranspose2d(256, 96, kernel_size=3, stride=2)
        self.dec3 = nn.ConvTranspose2d(96, 3, kernel_size=11, stride=4)

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.float().cuda()
        else:
            x = x.float()

        # encoder
        x = F.relu(self.bn1(self.enc1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.enc2(x)))
        x = self.pool(x)
        x = F.relu(self.enc3(x))  # the latent space representation

        # decoder
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        # x = F.relu(self.dec4(x))
        # x = F.sigmoid(self.out(x))

        return x
