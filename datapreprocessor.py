import pickle

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

        return data_loader_list

    @staticmethod
    def prepare_data_loader_test_10_splits(texture_test_data_set_path, texture_test_label_set_path,
                                           device):
        data_loader_list = []
        for i in range(10):
            idx = i + 1
            print("Split: {0}".format(idx))
            texture_test_data_set_path = texture_test_data_set_path.format(idx)
            texture_test_label_set_path = texture_test_label_set_path.format(idx)

            dL = DataLoader()
            texture_test_set, test_set_size = dL.get_tensor_set(texture_test_data_set_path,
                                                                texture_test_label_set_path,
                                                                device)
            print("Test set size: {0}".format(test_set_size))

            test_data_loader = torch.utils.data.DataLoader(texture_test_set, num_workers=1, shuffle=False,
                                                           pin_memory=True)

            data_loader_list.append(test_data_loader)

        return data_loader_list

    @staticmethod
    def preprocess_image_net(image_net_data_set_path, image_net_label_set_path, image_net_batch_size,
                             image_net_test_path, num_workers, split_size, device, type):
        print('-' * 50)
        print("{0} Statistics: ".format(type))
        print('-' * 50)
        dL = DataLoader()
        image_net_train_set, image_net_train_set_size, image_net_val_set, image_net_val_set_size, \
        image_net_test_set, image_net_test_size = \
            dL.get_test_train_val(
                image_net_data_set_path,
                image_net_label_set_path,
                split_size,
                device)

        print("Train set size: {0}".format(image_net_train_set_size))
        print("Val set size: {0}".format(image_net_val_set_size))
        print("Test set size: {0}".format(image_net_test_size))

        # pickle_out = open("./Dataset/ImageNet/ImageNet_Test.pickle", "wb")
        pickle_out = open(image_net_test_path, "wb")
        pickle.dump(image_net_test_set, pickle_out, protocol=4)
        pickle_out.close()
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

    # autoencoder dataloader
    # DTD
    @staticmethod
    def prepare_DTD_data_loader_test_10_splits(texture_test_data_set_path, texture_test_label_set_path,
                                               device):
        data_loader_list = []
        for i in range(10):
            idx = i + 1
            print("Split: {0}".format(idx))
            texture_test_data_set_path = texture_test_data_set_path.format(idx)
            texture_test_label_set_path = texture_test_label_set_path.format(idx)

            dL = DataLoader()
            texture_test_set, test_set_size = dL.get_tensor_set(texture_test_data_set_path,
                                                                texture_test_label_set_path,
                                                                device)
            print("Test set size: {0}".format(test_set_size))

            test_data_loader = torch.utils.data.DataLoader(texture_test_set, num_workers=1, shuffle=False,
                                                           pin_memory=True)

            data_loader_list.append(test_data_loader)

        return data_loader_list

    # autoencoder dataloader
    # Imagenet
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

    @staticmethod
    def preprocess_texture_except_DTD(texture_train_data_set_path, texture_train_label_set_path, batch_size,
                                      num_workers, device, split_size, type):
        data_loader_train_val_list = []
        data_loader_test_list = []
        # do this for 10 times
        for i in range(10):
            idx = i + 1
            print('-' * 50)
            print("{0} Statistics: ".format(type))
            print('-' * 50)
            dL = DataLoader()
            texture_train_set, texture_train_set_size, texture_val_set, texture_val_set_size, \
            texture_test_set, texture_test_size = \
                dL.get_test_train_val(
                    texture_train_data_set_path,
                    texture_train_label_set_path,
                    split_size,
                    device)

            print("Train set size: {0}".format(texture_train_set_size))
            print("Val set size: {0}".format(texture_val_set_size))
            print("Test set size: {0}".format(texture_test_size))
            texture_train_data_loader = torch.utils.data.DataLoader(texture_train_set,
                                                                    batch_size=batch_size,
                                                                    shuffle=True,
                                                                    num_workers=num_workers)

            texture_val_data_loader = torch.utils.data.DataLoader(
                texture_val_set, batch_size=batch_size, num_workers=1, shuffle=False, pin_memory=True)

            texture_test_data_loader = torch.utils.data.DataLoader(
                texture_test_set, batch_size=batch_size, num_workers=1, shuffle=False, pin_memory=True)

            data_loader_dict = {
                "train": texture_train_data_loader,
                "val": texture_val_data_loader
            }

            data_loader_train_val_list.append(data_loader_dict)
            data_loader_test_list.append(texture_test_data_loader)

        return data_loader_train_val_list, data_loader_test_list
