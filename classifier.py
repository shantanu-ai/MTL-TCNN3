import random

import torch.utils.data
from torch import nn, optim

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

    model_path_bn = "./Models/Auto_encoder_Model_epoch_300_lr_0.001_noise_factor_0.5.pt"
    device = Util.get_device()
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load(model_path_bn, map_location=device))

    split_size = 0.05
    train_parameters = {
        "epochs": 200,
        "learning_rate": 0.001,
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

    dL = DataLoader()
    texture_train_set, texture_val_set = dL.get_texture_train_set(texture_data_set_path, texture_label_set_path,
                                                                  split_size,
                                                                  device)
    texture_set_size = dL.get_texture_set_size()
    print(texture_set_size)

    # train(net, train_set, train_parameters, texture_set_size, device)

    image_net_data_set_path = "./Dataset/ImageNet/ImageNet_X.pickle"
    image_net_label_set_path = "./Dataset/ImageNet/ImageNet_Y.pickle"

    image_net_train_set, image_net_val_set = dL.get_image_net_train_set(image_net_data_set_path,
                                                                        image_net_label_set_path,
                                                                        split_size,
                                                                        device)
    image_net_set_size = dL.get_image_net_set_size()
    print(image_net_set_size)
    saved_model_name = "./Models/Classifier_Model_epoch_" + str(train_parameters["epochs"]) + "_lr_" + str(
        train_parameters["learning_rate"]) + ".pt"
    network = train(net, texture_train_set, image_net_train_set, train_parameters, image_net_set_size, texture_set_size,
                    device)
    torch.save(network.state_dict(), saved_model_name)
    print('Saved model parameters to disk.')


def train(network, texture_train_set, image_net_train_set, train_parameters, image_net_set_size, texture_set_size,
          device):
    print(device)
    print("..Training started..")
    epochs = train_parameters["epochs"]
    batch_size = train_parameters["batch_size"]
    lr = train_parameters["learning_rate"]
    num_workers = 0

    # set batch size
    texture_data_loader = torch.utils.data.DataLoader(texture_train_set,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      num_workers=num_workers)
    image_net_data_loader = torch.utils.data.DataLoader(image_net_train_set,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        num_workers=num_workers)

    i = 0
    batch_set = []
    task = ["Object_detection", "Texture_classification"]
    for image_net_data in image_net_data_loader:
        batch_set.append({task[0]: image_net_data})

    for texture_data in texture_data_loader:
        batch_set.append({task[1]: texture_data})

    # set optimizer - Adam
    optimizer = optim.Adam(network.parameters(), lr=lr)
    criterion = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss()]

    # start training
    for epoch in range(epochs):
        total_loss = 0
        object_corrects = 0
        texture_corrects = 0

        # shuffle all the mini batches
        random.shuffle(batch_set)

        for batch in batch_set:
            for key, value in batch.items():
                images, label = value
                images = images.to(device)
                label = label.to(device)

                # print("label: " + str(label.data))
                # zero out grads for every new iteration
                optimizer.zero_grad()

                # forward propagation
                outputs = network(images)

                # estimate texture loss
                if key == task[0]:
                    # Object Detection
                    object_pred = outputs[0]
                    loss = criterion[0](object_pred, label).to(device)
                elif key == task[1]:
                    # Texture classification
                    texture_pred = outputs[1]
                    loss = criterion[1](texture_pred, label).to(device)

                # back propagation
                loss.backward()

                # update weights
                # w = w - lr * grad_dw
                optimizer.step()

                total_loss += loss.item()

                if key == task[0]:
                    object_corrects += get_num_correct(outputs[0], label)
                elif key == task[1]:
                    texture_corrects += get_num_correct(outputs[1], label)

        object_corrects_accuracy = object_corrects / image_net_set_size
        texture_corrects_accuracy = texture_corrects / texture_set_size
        print(
            "epoch: {0}, loss: {1}, object_correct: {2},texture_correct: {3}, object accuracy: {4}, "
            "texture accuracy: {5}".format(epoch, total_loss, object_corrects, texture_corrects,
                                           object_corrects_accuracy,
                                           texture_corrects_accuracy))
    return network


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


main()
