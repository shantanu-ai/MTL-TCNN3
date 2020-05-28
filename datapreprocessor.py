import torch
from torch.utils.data import ConcatDataset

from dataLoader import DataLoader


class DataPreProcessor:
    @staticmethod
    def preprocess_DTD_train_val_10_splits(texture_train_data_set_path, texture_train_label_set_path,
                                           texture_val_data_set_path, texture_val_label_set_path,
                                           texture_batch_size, num_workers, device):
        print('-' * 50)
        print("DTD Statistics: ")
        print('-' * 50)
        data_loader_list = []
        for i in range(10):
            idx = i + 1
            print("Split: {0}".format(idx))
            texture_train_data_set_path = texture_train_data_set_path.format(idx)
            texture_train_label_set_path = texture_train_label_set_path.format(idx)
            texture_val_data_set_path = texture_val_data_set_path.format(idx)
            texture_val_label_set_path = texture_val_label_set_path.format(idx)

            dL = DataLoader()
            texture_train_set, train_set_size = dL.get_tensor_set(texture_train_data_set_path,
                                                                  texture_train_label_set_path,
                                                                  device)
            texture_val_set, val_set_size = dL.get_tensor_set(texture_val_data_set_path,
                                                              texture_val_label_set_path,
                                                              device)
            print("Train set size: {0}".format(train_set_size))
            print("Val set size: {0}".format(val_set_size))

            texture_train_data_loader = torch.utils.data.DataLoader(texture_train_set,
                                                                    batch_size=texture_batch_size,
                                                                    shuffle=True,
                                                                    num_workers=num_workers)
            texture_val_data_loader = torch.utils.data.DataLoader(
                texture_val_set, batch_size=texture_batch_size, num_workers=1, shuffle=False, pin_memory=True)

            data_loader_dict = {
                "train": texture_train_data_loader,
                "val": texture_val_data_loader
            }
            data_loader_list.append(data_loader_dict)
            break

        return data_loader_list

    @staticmethod
    def preprocess_image_net(image_net_data_set_path, image_net_label_set_path, image_net_batch_size,
                             split_size, num_workers, device):
        print('-' * 50)
        print("ImageNet Statistics: ")
        print('-' * 50)
        dL = DataLoader()
        image_net_train_set, image_net_train_set_size, image_net_val_set, image_net_val_set_size = \
            dL.get_image_net_train_set(
                image_net_data_set_path,
                image_net_label_set_path,
                split_size,
                device)

        print("Train set size: {0}".format(image_net_train_set_size))
        print("Val set size: {0}".format(image_net_val_set_size))
        image_net_train_data_loader = torch.utils.data.DataLoader(image_net_train_set,
                                                                  batch_size=image_net_batch_size,
                                                                  shuffle=True,
                                                                  num_workers=num_workers)
        image_net_val_data_loader = torch.utils.data.DataLoader(
            image_net_val_set, batch_size=image_net_batch_size,
            num_workers=1, shuffle=False, pin_memory=True)

        data_loader_dict = {
            "train": image_net_train_data_loader,
            "val": image_net_val_data_loader
        }

        return data_loader_dict

    @staticmethod
    def preprocess_autoencoder_train_val_10_splits(texture_train_data_set_path,
                                                   texture_train_label_set_path,
                                                   image_net_data_set_path,
                                                   image_net_label_set_path,
                                                   texture_batch_size,
                                                   num_workers,
                                                   device):
        print('-' * 50)
        print("ImageNet Statistics: ")
        print('-' * 50)
        dL = DataLoader()
        image_net_train_set, image_net_train_set_size = dL.split_train_test_validation_set_image_net(
            image_net_data_set_path,
            image_net_label_set_path,
            device)
        print("Train set size: {0}".format(image_net_train_set_size))
        data_loader_list = []

        print('-' * 50)
        print("DTD Statistics: ")
        print('-' * 50)
        for i in range(10):
            idx = i + 1
            print("Split: {0}".format(idx))
            texture_train_data_set_path = texture_train_data_set_path.format(idx)
            texture_train_label_set_path = texture_train_label_set_path.format(idx)

            dL = DataLoader()
            texture_train_set, train_set_size = dL.get_tensor_set(texture_train_data_set_path,
                                                                  texture_train_label_set_path,
                                                                  device)

            print("Train set size: {0}".format(train_set_size))
            train_set = ConcatDataset([image_net_train_set, texture_train_set])

            data_loader = torch.utils.data.DataLoader(train_set,
                                                      batch_size=texture_batch_size,
                                                      shuffle=True,
                                                      num_workers=num_workers)

            data_loader_list.append(data_loader)

        return data_loader_list
