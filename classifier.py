import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from MTLCNN import MTLCNN
from Util import Util
from autoEncoder import Autoencoder
from dataLoader import DataLoader


def main():
    TEXTURE_LABELS = ["banded", "blotchy", "braided", "bubbly", "bumpy", "chequered", "cobwebbed", "cracked",
                      "crosshatched", "crystalline",
                      "dotted", "fibrous", "flecked", "freckled", "frilly", "gauzy", "grid", "grooved", "honeycombed",
                      "interlaced", "knitted",
                      "lacelike", "lined", "marbled", "matted", "meshed", "paisley", "perforated", "pitted", "pleated",
                      "polka-dotted", "porous",
                      "potholed", "scaly", "smeared", "spiralled", "sprinkled", "stained", "stratified", "striped",
                      "studded", "swirly", "veined",
                      "waffled", "woven", "wrinkled", "zigzagged"]

    IMAGE_NET_LABELS = ["alp", "artichoke", "Band Aid", "bathing cap", "bookshop", "butcher shop", "carbonara", "chain",
                        "chainlink fence",
                        "cheetah", "cliff dwelling", "common iguana", "confectionery", "corn", "dishrag", "dock",
                        "flat-coated retriever",
                        "gibbon", "grocery store", "head cabbage", "honeycomb", "jigsaw puzzle", "lakeside",
                        "miniature poodle", "orangutan",
                        "partridge", "rapeseed", "sandbar", "sea urchin", "shoe shop", "shower curtain", "stone wall",
                        "theater curtain",
                        "tile roof", "vault", "velvet", "window screen", "wool"]

    model_path_bn = "./Models/Auto_encoder_Model_epoch_100_lr_0.001_noise_factor_0.5.pt"
    device = Util.get_device()
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load(model_path_bn, map_location=device))

    split_size = 0.05
    train_parameters = {
        "epochs": 1,
        "learning_rate": 0.001,
        "noise_factor": 0.5,
        "batch_size": 32
    }

    init_weights = {
        "conv1_wt": model.enc1.weight.data,
        "conv1_bias": model.enc1.bias.data,
        "conv2_wt": model.enc2.weight.data,
        "conv2_bias": model.enc2.bias.data,
        "conv3_wt": model.enc3.weight.data,
        "conv3_bias": model.enc3.bias.data
    }

    net = MTLCNN(init_weights, TEXTURE_LABELS, IMAGE_NET_LABELS, device).to(device)

    # texture classification test
    texture_data_set_path = "./Dataset/Texture/DTD/Texture_DTD_Train_X.pickle"
    texture_label_set_path = "./Dataset/Texture/DTD/Texture_DTD_Train_Y.pickle"

    # dL = DataLoader()
    # train_set, val_set = dL.get_texture_train_set(texture_data_set_path, texture_label_set_path,
    #                                               split_size,
    #                                               device)
    # texture_set_size = dL.get_texture_set_size()
    # print(texture_set_size)

    # train(net, train_set, train_parameters, texture_set_size, device)

    image_net_data_set_path = "./Dataset/ImageNet/ImageNet_X.pickle"
    image_net_label_set_path = "./Dataset/ImageNet/ImageNet_Y.pickle"

    dL = DataLoader()
    train_set, val_set = dL.get_image_net_train_set(image_net_data_set_path, image_net_label_set_path,
                                                    split_size,
                                                    device)
    image_net_set_size = dL.get_image_net_set_size()
    print(image_net_set_size)
    train(net, train_set, train_parameters, image_net_set_size, device)


def train(network, train_set, train_parameters, texture_set_size, device):
    print(device)
    print("..Training started..")
    epochs = train_parameters["epochs"]
    batch_size = train_parameters["batch_size"]
    lr = train_parameters["learning_rate"]
    noise_factor = train_parameters["noise_factor"]
    num_workers = 64

    # set batch size
    data_loader = torch.utils.data.DataLoader(train_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)

    # set optimizer - Adam
    optimizer = optim.Adam(network.parameters(), lr=lr)
    criterion = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss()]

    # start training
    for epoch in range(epochs):
        total_loss = 0
        texture_corrects = 0

        for batch in data_loader:
            images, label = batch
            images = images.to(device)
            label = label.to(device)
            # print("label: " + str(label.data))
            # zero out grads for every new iteration
            optimizer.zero_grad()

            # forward propagation
            outputs = network(images)
            # estimate texture loss
            loss_gender = criterion[1](outputs, label).to(device)

            # back propagation
            loss_gender.backward()

            # update weights
            # w = w - lr * grad_dw
            optimizer.step()

            total_loss += loss_gender.item()
            texture_corrects += get_num_correct(outputs, label)
            # break
        texture_corrects_accuracy = texture_corrects / texture_set_size
        print("epoch: {0}, loss: {1}, total_correct: {2} accuracy: {3}".
              format(epoch, total_loss, texture_corrects, texture_corrects_accuracy))
        # break
    return network


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


main()
