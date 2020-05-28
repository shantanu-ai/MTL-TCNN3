import pickle

import torch
import matplotlib.pyplot as plt

def test_pickle():
    # print("Test")
    # model_path_bn = "./Models/Auto_encoder_Model_epoch_300_lr_0.001_noise_factor_0.5_split_{0}.pt"
    # model_path_bn = model_path_bn.format(1)
    # print(model_path_bn)

    pickle_in = open("/Users/shantanughosh/Desktop/Shantanu_MS/Research/Dapeng_Wu/Git_Hub/Texture_Classification/Dataset/ImageNet/ImageNet_X.pickle", "rb")
    X = pickle.load(pickle_in)
    print(X.shape)
    print(X[89].shape)
    X = X.swapaxes(1, 3)
    plt.imshow(X[89])
    plt.show()


# create_pickle_for_training()

test_pickle()

