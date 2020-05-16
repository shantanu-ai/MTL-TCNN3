import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle


class Data_Preprocess_Manager:
    def create_training_data(self, data_dir, labels, img_size):
        training_data = []
        for label in labels:
            path = os.path.join(data_dir, label)
            class_num = labels.index(label)
            for img in os.listdir(path):
                try:
                    img_path = os.path.join(path, img)
                    img_array = cv2.imread(img_path)
                    resized_img = cv2.resize(img_array, (img_size, img_size))
                    # plt.imshow(resized_img)
                    # plt.show()
                    print(img_path)
                    training_data.append([resized_img, class_num])

                except Exception as e:
                    pass

            # shuffle the training data
            random.shuffle(training_data)

        return training_data

    @staticmethod
    def create_pickle_file(training_data, x_pickle_file_name, y_pickle_file_name):
        X = []
        Y = []
        for data, label in training_data:
            X.append(data)
            Y.append(label)

        X = np.array(X)
        X = X.swapaxes(1, 3)
        print(X.shape)
        print(len(Y))

        # Create X
        # print(x_pickle_file_name)
        # pickle_out = open(x_pickle_file_name, "wb")
        # pickle.dump(X, pickle_out)
        # pickle_out.close()

        # Create Y
        # print(y_pickle_file_name)
        # pickle_out = open(y_pickle_file_name, "wb")
        # pickle.dump(Y, pickle_out)
        # pickle_out.close()


def create_pickle_for_training():
    # ImageNet Dataset
    IMAGE_DATA_DIR = "/Users/shantanughosh/Desktop/Shantanu_MS/Research/Texture_Classfication/data_root_folder/dataset/imagenet/imagenet_images"
    IMAGE_NET_LABELS = ["alp", "artichoke", "Band Aid", "bathing cap", "bookshop", "butcher shop", "carbonara", "chain",
                        "chainlink fence",
                        "cheetah", "cliff dwelling", "common iguana", "confectionery", "corn", "dishrag", "dock",
                        "flat-coated retriever",
                        "gibbon", "grocery store", "head cabbage", "honeycomb", "jigsaw puzzle", "lakeside",
                        "miniature poodle", "orangutan",
                        "partridge", "rapeseed", "sandbar", "sea urchin", "shoe shop", "shower curtain", "stone wall",
                        "theater curtain",
                        "tile roof", "vault", "velvet", "window screen", "wool"]

    # Texture Dataset
    # DTD
    TEXTURE_DATA_DIR = "/Users/shantanughosh/Desktop/Shantanu_MS/Research/Texture_Classfication/data_root_folder/dataset/texture/dtd/images"
    TEXTURE_LABELS = ["banded", "blotchy", "braided", "bubbly", "bumpy", "chequered", "cobwebbed", "cracked",
                      "crosshatched", "crystalline",
                      "dotted", "fibrous", "flecked", "freckled", "frilly", "gauzy", "grid", "grooved", "honeycombed",
                      "interlaced", "knitted",
                      "lacelike", "lined", "marbled", "matted", "meshed", "paisley", "perforated", "pitted", "pleated",
                      "polka-dotted", "porous",
                      "potholed", "scaly", "smeared", "spiralled", "sprinkled", "stained", "stratified", "striped",
                      "studded", "swirly", "veined",
                      "waffled", "woven", "wrinkled", "zigzagged"]
    # TEXTURE_LABELS = ["banded"]

    image_size = 227
    #
    dpm = Data_Preprocess_Manager()
    image_net_training_data = dpm.create_training_data(IMAGE_DATA_DIR, IMAGE_NET_LABELS, 227)

    texture_training_data = dpm.create_training_data(TEXTURE_DATA_DIR, TEXTURE_LABELS, 227)

    for sample in texture_training_data[:10]:
        print(sample[1])

    x_pickle_file = "ImageNet_X.pickle"
    y_pickle_file = "ImageNet_Y.pickle"
    dpm.create_pickle_file(image_net_training_data, x_pickle_file, y_pickle_file)

    x_pickle_file = "Texture_DTD_X.pickle"
    y_pickle_file = "Texture_DTD_Y.pickle"
    dpm.create_pickle_file(texture_training_data, x_pickle_file, y_pickle_file)


create_pickle_for_training()



