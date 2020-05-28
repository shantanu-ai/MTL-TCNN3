import torch.utils.data
from torch import nn, optim

from Alexnet_single import Alex_netCNN
from Util import Util
from dataLoader import DataLoader


def main():
    device = Util.get_device()
    print(device)
    IMAGE_NET_LABELS = ["alp", "artichoke", "Band Aid", "bathing cap", "bookshop", "butcher shop", "carbonara", "chain",
                        "chainlink fence",
                        "cheetah", "cliff dwelling", "common iguana", "confectionery", "corn", "dishrag", "dock",
                        "flat-coated retriever",
                        "gibbon", "grocery store", "head cabbage", "honeycomb", "jigsaw puzzle", "lakeside",
                        "miniature poodle", "orangutan",
                        "partridge", "rapeseed", "sandbar", "sea urchin", "shoe shop", "shower curtain", "stone wall",
                        "theater curtain",
                        "tile roof", "vault", "velvet", "window screen", "wool"]

    print("Texture_label: " + str(len(IMAGE_NET_LABELS)))

    split_size = 0.05
    train_parameters = {
        "epochs": 100,
        "learning_rate": 0.001,
        "texture_batch_size": 32,
        "image_net_batch_size": 256
    }

    net = Alex_netCNN(IMAGE_NET_LABELS, device).to(device)

    dL = DataLoader()
    image_net_data_set_path = "./Dataset/ImageNet/ImageNet_X.pickle"
    image_net_label_set_path = "./Dataset/ImageNet/ImageNet_Y.pickle"

    image_net_train_set, image_net_val_set = dL.get_image_net_train_set(image_net_data_set_path,
                                                                        image_net_label_set_path,
                                                                        split_size,
                                                                        device)
    image_net_train_size = dL.get_image_net_train_size()
    image_net_val_size = dL.get_image_net_test_size()
    print(image_net_train_size)
    print(image_net_val_size)
    saved_model_name = "./Models/SingleAlex_net_Classifier_Model_epoch_" + str(train_parameters["epochs"]) \
                       + "_lr_" + str(
        train_parameters["learning_rate"]) + ".pt"

    network = train(net, image_net_train_set, train_parameters, image_net_train_size,
                    device)

    torch.save(network.state_dict(), saved_model_name)
    test(network, image_net_val_set, image_net_val_size, device)

    # logistics = NeuralNetClassifier(net, max_epochs=10, lr=0.001, device=device)
    # scores = cross_val_score(logistics, train_data_set, labels_set, cv=10, scoring="accuracy")
    # print('score: ' + str(scores.mean()))

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


def test(network, image_net_val_set, image_net_val_size,
         device):
    print(device)
    print("..Testing started..")
    num_workers = 0

    data_loader = torch.utils.data.DataLoader(
        image_net_val_set, num_workers=1, shuffle=False, pin_memory=True)
    # start
    corrects = 0

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
        corrects += get_num_correct(outputs, label)

    corrects_accuracy = corrects / image_net_val_size
    print("total:{0} texture accuracy: {1}".format(corrects, corrects_accuracy))

    return network


def train(network, image_net_train_set, train_parameters, image_net_set_size,
          device):
    print(device)
    print("..Training started..")
    epochs = train_parameters["epochs"]
    texture_batch_size = train_parameters["texture_batch_size"]
    image_net_batch_size = train_parameters["image_net_batch_size"]
    lr = train_parameters["learning_rate"]
    num_workers = 0

    # set batch size
    image_net_data_loader = torch.utils.data.DataLoader(image_net_train_set,
                                                        batch_size=image_net_batch_size,
                                                        shuffle=True,
                                                        num_workers=num_workers)

    task = ["Object_detection", "Texture_classification"]
    # set optimizer - Adam
    optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()

    # start training
    for epoch in range(epochs):
        total_loss = 0
        object_corrects = 0
        for value in image_net_data_loader:
            images, label = value
            images = images.to(device)
            label = label.to(device)

            # print("label: " + str(label.data))
            # zero out grads for every new iteration
            optimizer.zero_grad()

            # forward propagation
            outputs = network(images)
            loss = criterion(outputs, label).to(device)

            # back propagation
            loss.backward()

            # update weights
            # w = w - lr * grad_dw
            optimizer.step()

            total_loss += loss.item()

            object_corrects += get_num_correct(outputs, label)

        object_corrects_percent = object_corrects / image_net_set_size
        print(
            "epoch: {0}, loss: {1}, object_correct: {2} "
            "texture accuracy: {3}".format(epoch, total_loss, object_corrects,
                                           object_corrects_percent))
    return network


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


main()
