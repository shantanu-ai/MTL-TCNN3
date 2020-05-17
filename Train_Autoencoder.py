import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from autoEncoder import Autoencoder


class Train_Auto_encoder:
    def train_auto_encoder(self, train_set, train_parameters, saved_model_name, device):
        model = self.__train_model(train_set, train_parameters, saved_model_name, device)
        print("..Training completed..")
        return model

    def __train_model(self, train_set, train_parameters, saved_model_name, device):
        model = self.__train(train_set, train_parameters, device)
        torch.save(model.state_dict(), saved_model_name)
        print('Saved model parameters to disk.')
        return model

    @staticmethod
    def __train(train_set, train_parameters, device):
        print("..Training started..")
        network = Autoencoder().to(device)
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
        criterion = nn.MSELoss()

        # start training
        for epoch in range(epochs):
            total_loss = 0
            total_correct = 0

            for batch in data_loader:
                images, _ = batch

                noisy_image = images + noise_factor * torch.randn(*images.shape)
                noisy_image = np.clip(noisy_image, 0., 1.)

                images = images.to(device)
                noisy_image = noisy_image.to(device)

                # zero out grads for every new iteration
                optimizer.zero_grad()

                # forward propagation
                outputs = network(noisy_image)
                # estimate loss
                loss = criterion(outputs, images)

                # back propagation
                loss.backward()

                # update weights
                # w = w - lr * grad_dw
                optimizer.step()

                total_loss += loss.item()
            print("epoch: {0}, loss: {1}".format(epoch, total_loss))
        return network
