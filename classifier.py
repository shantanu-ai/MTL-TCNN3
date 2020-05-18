import torch.utils.data

from MTLCNN import MTLCNN
from Util import Util
from autoEncoder import Autoencoder


device = Util.get_device()
model_path_bn = "./Models/Auto_encoder_Model_epoch_100_lr_0.001_noise_factor_0.5.pt"
device = Util.get_device()
model = Autoencoder().to(device)
model.load_state_dict(torch.load(model_path_bn, map_location=device))

init_weights = {
    "conv1_wt": model.enc1.weight.data,
    "conv1_bias": model.enc1.bias.data,
    "conv2_wt": model.enc2.weight.data,
    "conv2_bias": model.enc2.bias.data,
    "conv3_wt": model.enc3.weight.data,
    "conv3_bias": model.enc3.bias.data
}

net = MTLCNN(init_weights, device)

print(net)
