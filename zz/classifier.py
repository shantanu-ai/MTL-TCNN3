import torch.utils.data
from torch import nn, optim

from MTLCNN_single import MTLCNN_single
from Util import Util
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

    print("Texture_label: " + str(len(TEXTURE_LABELS)))
    model_path_bn = "./Models/Auto_encoder_Model_epoch_300_lr_0.001_noise_factor_0.5.pt"

    device = Util.get_device()
    print(device)
    # model = Autoencoder().to(device)
    # model.load_state_dict(torch.load(model_path_bn, map_location=device))

    # split_size = 0.05

    # init_weights = {
    #     "conv1_wt": model.enc1.weight.data,
    #     "conv1_bias": model.enc1.bias.data,
    #     "conv2_wt": model.enc2.weight.data,
    #     "conv2_bias": model.enc2.bias.data,
    #     "conv3_wt": model.enc3.weight.data,
    #     "conv3_bias": model.enc3.bias.data
    # }

    train_parameters = {
        "epochs": 100,
        "learning_rate": 0.001,
        "texture_batch_size": 32,
        "image_net_batch_size": 256
    }

    net = MTLCNN_single(TEXTURE_LABELS).to(device)

    texture_train_data_set_path = "./Dataset/Texture/DTD/Texture_DTD_train2_X.pickle"
    texture_train_label_set_path = "./Dataset/Texture/DTD/Texture_DTD_train2_Y.pickle"

    texture_val_data_set_path = "./Dataset/Texture/DTD/Texture_DTD_val2_X.pickle"
    texture_val_label_set_path = "./Dataset/Texture/DTD/Texture_DTD_val2_Y.pickle"

    dL = DataLoader()
    texture_train_set, train_set_size = dL.get_tensor_set(texture_train_data_set_path,
                                                          texture_train_label_set_path,
                                                          device)
    texture_val_set, val_set_size = dL.get_tensor_set(texture_val_data_set_path,
                                                      texture_val_label_set_path,
                                                      device)

    print("Train set size: {0}".format(train_set_size))
    print("Val set size: {0}".format(val_set_size))

    saved_model_name = "./Models/Texture_Single_Classifier_Model_epoch_" + str(
        train_parameters["epochs"]) + "_lr_" + str(
        train_parameters["learning_rate"]) + ".pt"

    train_arguments = {
        "net": net,
        "texture_train_set": texture_train_set,
        "train_set_size": train_set_size,
        "texture_val_set": texture_val_set,
        "val_set_size": val_set_size,
        "train_parameters": train_parameters
    }
    network = train(train_arguments,
                    device)
    #
    # torch.save(network.state_dict(), saved_model_name)
    # test(network, image_net_val_set)

    print('Saved model parameters to disk.')

    # test
    # texture_data_set_path = "./Dataset/Texture/DTD/Texture_DTD_Test_X.pickle"
    # texture_label_set_path = "./Dataset/Texture/DTD/Texture_DTD_Test_Y.pickle"
    # dL = DataLoader()
    # texture_test_set = dL.pre_process_test_texture(texture_data_set_path, texture_label_set_path, device)
    #
    # model_path_bn = "./Models/Single_Classifier_Model_epoch_1000_lr_0.001.pt"
    # device = Util.get_device()
    # network_model = MTLCNN_single(init_weights, TEXTURE_LABELS, IMAGE_NET_LABELS, device).to(device)
    # network_model.load_state_dict(torch.load(model_path_bn, map_location=device))
    #
    # texture_set_size = dL.get_texture_set_size()
    # print(texture_set_size)
    # test(network_model, texture_test_set, texture_set_size, device)


def test(network, texture_test_set, texture_set_size,
         device):
    print(device)
    print("..Testing started..")
    num_workers = 0

    data_loader = torch.utils.data.DataLoader(
        texture_test_set, num_workers=1, shuffle=False, pin_memory=True)
    # start
    texture_corrects = 0

    for batch in data_loader:
        images, label = batch
        images = images.to(device)
        label = label.to(device)

        # print("label: " + str(label.data))
        # zero out grads for every new iteration

        # forward propagation
        outputs = network(images)

        # estimate texture loss

        # Texture classification
        texture_corrects += get_num_correct(outputs, label)

    texture_corrects_accuracy = texture_corrects / texture_set_size
    print("total:{0} texture accuracy: {1}".format(texture_corrects, texture_corrects_accuracy))

    return network


def train(train_arguments, device):
    network = train_arguments["net"]
    texture_train_set = train_arguments["texture_train_set"]
    train_set_size = train_arguments["train_set_size"]
    texture_val_set = train_arguments["texture_val_set"]
    val_set_size = train_arguments["val_set_size"]
    train_parameters = train_arguments["train_parameters"]

    print("..Training started..")
    print(val_set_size)
    epochs = train_parameters["epochs"]
    texture_batch_size = train_parameters["texture_batch_size"]
    lr = train_parameters["learning_rate"]
    num_workers = 0

    # set batch size
    texture_train_data_loader = torch.utils.data.DataLoader(texture_train_set,
                                                            batch_size=texture_batch_size,
                                                            shuffle=True,
                                                            num_workers=num_workers)
    texture_val_data_loader = torch.utils.data.DataLoader(
        texture_val_set, num_workers=1, shuffle=False, pin_memory=True)

    # set optimizer - Adam
    optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()

    # start training
    for epoch in range(epochs):
        total_train_loss = 0
        total_val_loss = 0
        texture_train_corrects = 0
        texture_val_corrects = 0
        # training
        for value in texture_train_data_loader:
            train_images, train_label = value
            train_images = train_images.to(device)
            train_label = train_label.to(device)

            # print("label: " + str(label.data))
            # zero out grads for every new iteration
            optimizer.zero_grad()

            # forward propagation
            train_outputs = network(train_images)
            train_loss = criterion(train_outputs, train_label).to(device)

            # back propagation
            train_loss.backward()

            # update weights
            # w = w - lr * grad_dw
            optimizer.step()

            total_train_loss += train_loss.item()
            texture_train_corrects += get_num_correct(train_outputs, train_label)

        # validation
        for batch in texture_val_data_loader:
            val_images, val_label = batch
            val_images = val_images.to(device)
            val_label = val_label.to(device)

            # forward propagation
            val_outputs = network(val_images)
            val_loss = criterion(val_outputs, val_label).to(device)

            # estimate texture loss

            # Texture classification
            total_val_loss += val_loss.item()
            texture_val_corrects += get_num_correct(val_outputs, val_label)

        texture_train_accuracy = texture_train_corrects / train_set_size
        texture_val_accuracy = texture_val_corrects / val_set_size
        print("epoch: {0}, train_loss: {1}, val_loss: {2}, train_correct: {3}, val_correct: {4} "
              "train accuracy: {5}, val accuracy: {6}".format(epoch, total_train_loss, total_val_loss,
                                                              texture_train_corrects, texture_val_corrects,
                                                              texture_train_accuracy, texture_val_accuracy))
    return network


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


main()
