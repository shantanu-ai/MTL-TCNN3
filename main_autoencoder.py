from Train_Autoencoder import Train_Auto_encoder
from Util import Util
from dataLoader import DataLoader
from validate import validate_Auto_encoder

def auto_encoder_BL():
    device = Util.get_device()
    image_net_data_set_path = "./Dataset/ImageNet/ImageNet_X.pickle"
    image_net_label_set_path = "./Dataset/ImageNet/ImageNet_Y.pickle"
    texture_data_set_path = "./Dataset/Texture/DTD/Texture_DTD_Train_X.pickle"
    texture_label_set_path = "./Dataset/Texture/DTD/Texture_DTD_Train_Y.pickle"

    split_size = 0.05
    train_parameters = {
        "epochs": 300,
        "learning_rate": 0.001,
        "noise_factor": 0.5,
        "batch_size": 32
    }

    saved_model_name = "./Models/Auto_encoder_Model_epoch_" + str(train_parameters["epochs"]) + "_lr_" + str(
        train_parameters["learning_rate"]) + "_noise_factor_" + str(train_parameters["noise_factor"]) + ".pt"

    dL = DataLoader()
    train_set, val_set = dL.split_train_test_validation(image_net_data_set_path, image_net_label_set_path,
                                                        texture_data_set_path, texture_label_set_path,
                                                        split_size,
                                                        device)

    train = Train_Auto_encoder()
    model = train.train_auto_encoder(train_set, train_parameters, saved_model_name, device)

    task = "---validating"
    test = validate_Auto_encoder()
    test.validate_auto_encoder(val_set, model, task)


if __name__ == '__main__':
    auto_encoder_BL()
