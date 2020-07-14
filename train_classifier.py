import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import copy

from MTLCNN_single import MTLCNN_single
from MultitaskClassifier import MultitaskClassifier
from Util import Util


class Train_Classifier:
    def train_classifier(self, train_arguments, device, dataset_name=""):
        self.__train(train_arguments, device, dataset_name=dataset_name)

    def train_classifier_single(self, train_arguments, device):
        self.__train_single(train_arguments, device)

    def __train_single(self, train_arguments, device):
        TEXTURE_LABELS = train_arguments["TEXTURE_LABELS"]
        data_loader_list = train_arguments["texture_data_loader_list"]
        train_parameters = train_arguments["train_parameters"]
        saved_model_name = train_arguments["saved_model_name"]

        print("..Training started..")
        epochs = train_parameters["epochs"]
        lr = train_parameters["learning_rate"]
        phases = ['train', 'val']
        # set batch size

        # set optimizer - Adam

        split_id = 0

        # start training
        for data_loader_dict in data_loader_list:
            # initialise network for each dataset
            network = MTLCNN_single(TEXTURE_LABELS).to(device)
            optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=0.0005)
            criterion = nn.CrossEntropyLoss()
            min_correct = 0
            split_id += 1
            print('-' * 50)
            print("Split: {0} =======>".format(split_id))

            # start epoch
            for epoch in range(epochs):
                print('Epoch {}/{}'.format(epoch, epochs - 1))
                print('-' * 20)

                for phase in phases:
                    if phase == 'train':
                        network.train()  # Set model to training mode
                    else:
                        network.eval()  # Set model to evaluate mode

                    running_loss = 0
                    running_correct = 0
                    total_image_per_epoch = 0

                    for batch in data_loader_dict[phase]:
                        images, label = batch
                        images = images.to(device)
                        label = label.to(device)

                        optimizer.zero_grad()

                        output = network(images)
                        loss = criterion(output, label).to(device)
                        total_image_per_epoch += images.size(0)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                        running_loss += loss.item() * images.size(0) * 2
                        running_correct += Util.get_num_correct(output, label)

                    epoch_loss = running_loss / total_image_per_epoch

                    epoch_accuracy = running_correct / total_image_per_epoch
                    print(
                        "{0} ==> loss: {1}, correct: {2}/{3}, accuracy: {4}".format(phase, epoch_loss, running_correct,
                                                                                    total_image_per_epoch,
                                                                                    epoch_accuracy))
                    if phase == 'val' and running_correct > min_correct:
                        print("saving model with correct: {0}, improved over previous {1}"
                              .format(running_correct, min_correct))
                        min_correct = running_correct
                        best_model_wts = copy.deepcopy(network.state_dict())
                        torch.save(best_model_wts, saved_model_name.format(split_id))

    def __train(self, train_arguments, device, dataset_name=""):

        # labels
        IMAGE_NET_LABELS = train_arguments["IMAGE_NET_LABELS"]
        TEXTURE_LABELS = train_arguments["TEXTURE_LABELS"]

        # data loader
        image_net_data_loader_dict = train_arguments["image_net_data_loader_dict"]
        # image_net_T_data_loader_dict = train_arguments["image_net_T_data_loader_dict"]
        texture_data_loader_list = train_arguments["texture_data_loader_list"]

        train_parameters = train_arguments["train_parameters"]
        saved_model_name = train_arguments["saved_model_name"]

        print("..Training started..")

        epochs = train_parameters["epochs"]
        lr = train_parameters["learning_rate"]
        weight_decay = train_parameters["weight_decay"]
        labels = {
            "image_net_labels": IMAGE_NET_LABELS,
            # "image_net_labels_S2": IMAGE_NET_LABELS_S2,
            # "image_net_labels_T": IMAGE_NET_LABELS_T,
            "texture_labels": TEXTURE_LABELS
        }
        phases = ['train', 'val']
        split_id = 0
        # task = ["Object_detection", "Texture_classification"]
        task = ["Object_detection", "Texture_classification"]

        for texture_data_loader_dict in texture_data_loader_list:
            print("Dataset name: {0}".format(dataset_name))
            split_id += 1
            print('-' * 50)
            print("Split: {0} =======>".format(split_id))
            model_path = saved_model_name.format(split_id)
            print("Model: {0}".format(model_path))

            network = MultitaskClassifier(labels).to(device)

            optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss()]
            texture_min_val_correct = 0

            for epoch in range(epochs):
                print('Epoch {}/{}'.format(epoch, epochs - 1))
                print('-' * 20)
                for phase in phases:
                    print("Phase: " + phase)
                    if phase == 'train':
                        network.train()  # Set model to training mode
                    else:
                        network.eval()  # Set model to evaluate mode

                    running_loss = 0
                    running_loss_imagenet = 0
                    running_loss_texture = 0

                    running_imagenet_correct = 0
                    running_texture_correct = 0

                    total_imagenet_image_per_epoch = 0
                    total_texture_image_per_epoch = 0

                    batch_set = self.__get_batch_set(image_net_data_loader_dict,
                                                     texture_data_loader_dict, phase, task)

                    if phase == "train":
                        random.shuffle(batch_set)

                    for batch_dict in batch_set:
                        for task_id, batch in batch_dict.items():
                            images, label = batch
                            images = images.to(device)
                            label = label.to(device)

                            optimizer.zero_grad()

                            outputs = network(images)

                            if task_id == task[0]:
                                total_imagenet_image_per_epoch += images.size(0)
                                loss = criterion[0](outputs[0], label).to(device)
                                running_loss_imagenet += loss.item()
                                # network.conv4.weight.requires_grad = True
                                # network.conv4.bias.requires_grad = True
                                # network.conv5.weight.requires_grad = True
                                # network.conv5.bias.requires_grad = True
                                # network.object_detect_fc1.weight.requires_grad = True
                                # network.object_detect_fc1.bias.requires_grad = True
                                # network.object_detect_fc2.weight.requires_grad = True
                                # network.object_detect_fc2.bias.requires_grad = True
                                # network.object_detect_out.weight.requires_grad = True
                                # network.object_detect_out.bias.requires_grad = True
                                #
                                # network.fc_texture_1.weight.requires_grad = False
                                # network.fc_texture_1.bias.requires_grad = False
                                # network.fc_texture_2.weight.requires_grad = False
                                # network.fc_texture_2.bias.requires_grad = False
                                # network.texture_out.weight.requires_grad = False
                                # network.texture_out.bias.requires_grad = False

                            elif task_id == task[1]:
                                total_texture_image_per_epoch += images.size(0)
                                loss = criterion[1](outputs[1], label).to(device)
                                running_loss_texture += loss.item()

                                # network.conv4.weight.requires_grad = False
                                # network.conv4.bias.requires_grad = False
                                # network.conv5.weight.requires_grad = False
                                # network.conv5.bias.requires_grad = False
                                # network.object_detect_fc1.weight.requires_grad = False
                                # network.object_detect_fc1.bias.requires_grad = False
                                # network.object_detect_fc2.weight.requires_grad = False
                                # network.object_detect_fc2.bias.requires_grad = False
                                # network.object_detect_out.weight.requires_grad = False
                                # network.object_detect_out.bias.requires_grad = False
                                #
                                # network.fc_texture_1.weight.requires_grad = True
                                # network.fc_texture_1.bias.requires_grad = True
                                # network.fc_texture_2.weight.requires_grad = True
                                # network.fc_texture_2.bias.requires_grad = True
                                # network.texture_out.weight.requires_grad = True
                                # network.texture_out.bias.requires_grad = True

                            if phase == "train":
                                loss.backward()
                                optimizer.step()

                            running_loss += loss.item()

                            if task_id == task[0]:
                                running_imagenet_correct += Util.get_num_correct(outputs[0], label)
                            elif task_id == task[1]:
                                running_texture_correct += Util.get_num_correct(outputs[1], label)

                    epoch_loss = running_loss
                    epoch_loss_imagenet = running_loss_imagenet
                    epoch_loss_texture = running_loss_texture

                    epoch_imagenet_accuracy = running_imagenet_correct / total_imagenet_image_per_epoch
                    epoch_texture_accuracy = running_texture_correct / total_texture_image_per_epoch

                    print(
                        "{0} ==> loss: {1}, imagenet loss:{2}, texture loss:{3}, "
                        "imagenet correct: {4}/{5}, imagenet accuracy: {6} "
                        "texture correct: {7}/{8}, texture accuracy: {9}, ".format(phase,
                                                                                   epoch_loss,
                                                                                   epoch_loss_imagenet,
                                                                                   epoch_loss_texture,
                                                                                   running_imagenet_correct,
                                                                                   total_imagenet_image_per_epoch,
                                                                                   epoch_imagenet_accuracy,
                                                                                   running_texture_correct,
                                                                                   total_texture_image_per_epoch,
                                                                                   epoch_texture_accuracy
                                                                                   ))
                    if phase == 'val' and running_texture_correct > texture_min_val_correct:
                        print("saving model with correct: {0}, improved over previous {1}"
                              .format(running_texture_correct, texture_min_val_correct))
                        texture_min_val_correct = running_texture_correct
                        # saved_model_name = saved_model_name.format(split_id)
                        torch.save(network.state_dict(), model_path)

        print("Training ended")

    @staticmethod
    def __get_batch_set(image_net_data_loader_dict,
                        texture_data_loader_dict, phase, task):
        batch_set = []
        for image_net_S2_data in image_net_data_loader_dict[phase]:
            batch_set.append({task[0]: image_net_S2_data})

        for texture_data in texture_data_loader_dict[phase]:
            batch_set.append({task[1]: texture_data})
        return batch_set

    def __save_init_weights(self, network):
        np.save("./init_weights/enc1_weight.npy", network.enc1.weight.cpu().data.numpy())
        np.save("./init_weights/enc1_bias.npy", network.enc1.bias.cpu().data.numpy())
        np.save("./init_weights/enc2_weight.npy", network.enc2.weight.cpu().data.numpy())
        np.save("./init_weights/enc2_bias.npy", network.enc2.bias.cpu().data.numpy())
        np.save("./init_weights/enc3_weight.npy", network.enc3.weight.cpu().data.numpy())
        np.save("./init_weights/enc3_bias.npy", network.enc3.bias.cpu().data.numpy())

    # @staticmethod
    # def initialize_model(auto_encoder_model_path, dataset_labels, device):
    #     model = Autoencoder().to(device)
    #     model.load_state_dict(torch.load(auto_encoder_model_path, map_location=device))
    #     TEXTURE_LABELS = dataset_labels["TEXTURE_LABELS"]
    #     IMAGE_NET_LABELS = dataset_labels["IMAGE_NET_LABELS"]
    #     auto_encoder_model = Autoencoder().to(device)
    #     init_weights = {
    #         "conv1_wt": model.enc1.weight.data,
    #         "conv1_bias": model.enc1.bias.data,
    #         "conv2_wt": model.enc2.weight.data,
    #         "conv2_bias": model.enc2.bias.data,
    #         "conv3_wt": model.enc3.weight.data,
    #         "conv3_bias": model.enc3.bias.data
    #     }
    #     auto_encoder_model.load_state_dict(torch.load(auto_encoder_model_path, map_location=device))
    #     network = MTLCNN(init_weights, TEXTURE_LABELS, IMAGE_NET_LABELS, device).to(device)
    #     return network
