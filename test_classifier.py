import numpy as np
import torch
import torch.utils.data

from MultitaskClassifier import MultitaskClassifier
from Util import Util


class Test_Classifier:
    def test_classifier(self, test_parameters, IMAGE_NET_LABELS, device, dataset_name=""):
        return self.__testMTL(test_parameters, IMAGE_NET_LABELS, device, dataset_name)

    @staticmethod
    def __testMTL(test_parameters, IMAGE_NET_LABELS, device, dataset_name=""):
        data_loader_test_list = test_parameters["data_loader_test_list"]
        model_path_bn = test_parameters["model_path_bn"]
        TEXTURE_LABELS = test_parameters["TEXTURE_LABELS"]

        print(model_path_bn)
        print(device)
        print("..Testing started..")

        split_id = 0
        accuracy_list = []

        # start testing
        for data_loader in data_loader_test_list:
            print("Dataset name: {0}".format(dataset_name))
            split_id += 1

            print('-' * 50)
            print("Split: {0} =======>".format(split_id))
            model_path = model_path_bn.format(split_id)
            print("Model: {0}".format(model_path))
            labels = {
                "image_net_labels": IMAGE_NET_LABELS,
                "texture_labels": TEXTURE_LABELS
            }
            network_model = MultitaskClassifier(labels).to(device)
            network_model.load_state_dict(torch.load(model_path, map_location=device))
            network_model.eval()
            total_image_per_epoch = 0
            texture_corrects = 0

            for batch in data_loader:
                images, label = batch
                images = images.to(device)
                label = label.to(device)

                outputs = network_model(images)
                total_image_per_epoch += images.size(0)
                texture_corrects += Util.get_num_correct(outputs[1], label)

            texture_corrects_accuracy = texture_corrects / total_image_per_epoch
            accuracy_list.append(texture_corrects_accuracy)
            print("total:{0} texture accuracy: {1}".format(texture_corrects, texture_corrects_accuracy))

        accuracy_np = np.asarray(accuracy_list)
        print("Mean accuracy: {0}".format(np.mean(accuracy_np)))
        print("Testing ended..")
        return accuracy_list, np.mean(accuracy_np)
