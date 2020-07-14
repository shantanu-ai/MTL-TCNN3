from Util import Util
# from validate import validate_Auto_encoder
from datapreprocessor import DataPreProcessor
from zz.validate import validate_Auto_encoder


def auto_encoder_BL():
    device = Util.get_device()
    # device = torch.device('cuda')
    print(device)
    image_net_data_set_path = "./Dataset/ImageNet/ImageNet_X.pickle"
    image_net_label_set_path = "./Dataset/ImageNet/ImageNet_Y.pickle"

    texture_train_data_set_path = "./Dataset/Texture/DTD/Texture_DTD_train{0}_X.pickle"
    texture_train_label_set_path = "./Dataset/Texture/DTD/Texture_DTD_train{0}_Y.pickle"

    texture_val_data_set_path = "./Dataset/Texture/DTD/Texture_DTD_val{0}_X.pickle"
    texture_val_label_set_path = "./Dataset/Texture/DTD/Texture_DTD_val{0}_Y.pickle"

    split_size = 0.05
    train_parameters = {
        "epochs": 100,
        "learning_rate": 0.001,
        "noise_factor": 0.3,
        "batch_size": 64
    }

    saved_model_name = "./Models/autoencoder/Auto_encoder_Model_epoch_" + str(
        train_parameters["epochs"]) + "_lr_" + str(
        train_parameters["learning_rate"]) + "_split{0}.pt"

    # data_loader_list = DataPreProcessor.preprocess_autoencoder_train_val_10_splits(texture_train_data_set_path,
    #                                                                                texture_train_label_set_path,
    #                                                                                image_net_data_set_path,
    #                                                                                image_net_label_set_path,
    #                                                                                train_parameters[
    #                                                                                    "batch_size"],
    #                                                                                num_workers=0,
    #                                                                                device=device)
    #
    # train = Train_Auto_encoder()
    # model = train.train_auto_encoder(data_loader_list, train_parameters, saved_model_name, device)

    task = "---validating"
    texture_test_data_set_path = "./Dataset/Texture/DTD/Texture_DTD_test{0}_X.pickle"
    texture_test_label_set_path = "./Dataset/Texture/DTD/Texture_DTD_test{0}_Y.pickle"

    data_loader_test_list = DataPreProcessor.prepare_DTD_data_loader_test_10_splits(texture_test_data_set_path,
                                                                                    texture_test_label_set_path,
                                                                                    device)
    model_path_bn = "./Models/autoencoder/Auto_encoder_Model_epoch_100_lr_0.001_split{0}.pt"
    test = validate_Auto_encoder()
    test.validate_auto_encoder(data_loader_test_list, model_path_bn, task, device)


if __name__ == '__main__':
    auto_encoder_BL()
