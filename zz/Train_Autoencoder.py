import numpy as np
import torch
import torch.nn as nn
from torch import optim

from zz.autoEncoder import Autoencoder


class Train_Auto_encoder:
    def train_auto_encoder(self, data_loader_list, train_parameters, saved_model_name, device):
        model = self.__train(data_loader_list, train_parameters, saved_model_name, device)
        print("..Training completed..")
        return model

    def __train(self, data_loader_list, train_parameters, saved_model_name, device):
        print("..Training started..")

        epochs = train_parameters["epochs"]
        lr = train_parameters["learning_rate"]

        split_id = 0
        print("Size of data_loader_list: " + str(len(data_loader_list)))

        for data_loader in data_loader_list:
            network = Autoencoder()
            network.to(device)
            # self.__save_init_weights(network)

            optimizer = optim.Adam(network.parameters(), lr=lr)
            criterion = nn.MSELoss()

            split_id += 1
            print('-' * 50)
            print("Split: {0} =======>".format(split_id))
            model_path = saved_model_name.format(split_id)
            print("Model: {0}".format(model_path))
            min_loss = 1000000.0
            for epoch in range(epochs):
                total_loss = 0

                for batch in data_loader:
                    images, _ = batch
                    images = images.to(device)

                    # zero out grads for every new iteration
                    optimizer.zero_grad()

                    # forward propagation
                    outputs = network(images)
                    # estimate loss
                    loss = criterion(outputs, images)

                    # back propagation
                    loss.backward()

                    # update weights
                    optimizer.step()

                    total_loss += loss.item()
                print("epoch: {0}/{1}, loss: {2}".format(epoch, epochs - 1, total_loss))
                # print(total_loss < min_loss)
                if total_loss < min_loss:
                    print("saving model with loss {0}, over previous: {1}".format(total_loss, min_loss))
                    torch.save(network.state_dict(), model_path)
                    min_loss = total_loss

        print('Saved models\' parameters to disk.')
        return network

    def __save_init_weights(self, network):
        np.save("./init_weights/enc1_weight.npy", network.enc1.weight.cpu().data.numpy())
        np.save("./init_weights/enc1_bias.npy", network.enc1.bias.cpu().data.numpy())
        np.save("./init_weights/enc2_weight.npy", network.enc2.weight.cpu().data.numpy())
        np.save("./init_weights/enc2_bias.npy", network.enc2.bias.cpu().data.numpy())
        np.save("./init_weights/enc3_weight.npy", network.enc3.weight.cpu().data.numpy())
        np.save("./init_weights/enc3_bias.npy", network.enc3.bias.cpu().data.numpy())
