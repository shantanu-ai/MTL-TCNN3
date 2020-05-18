import torch.utils.data

from Util import Util
from autoEncoder import Autoencoder
from dataLoader import DataLoader
from validate import validate_Auto_encoder


def test_auto_encoder():
    model_path_bn = "./Models/Auto_encoder_Model_epoch_100_lr_0.001_noise_factor_0.5.pt"
    device = Util.get_device()
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load(model_path_bn, map_location=device))
    texture_data_set_path_test = "./Dataset/Texture/DTD/Texture_DTD_Train_X.pickle"

    dL = DataLoader()
    test_data_set = dL.pre_process_test(texture_data_set_path_test)

    task = "---testing"
    test = validate_Auto_encoder()
    test.validate_auto_encoder(test_data_set, model, task)


if __name__ == '__main__':
    test_auto_encoder()
