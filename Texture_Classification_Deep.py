import pandas as pd

from Util import Util
from datapreprocessor import DataPreProcessor
from test_classifier import Test_Classifier
from train_classifier import Train_Classifier


class Texture_Classification_Deep:
    @staticmethod
    def DTD_Train_test():
        device = Util.get_device()
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device)
        TEXTURE_LABELS = ["banded", "blotchy", "braided", "bubbly", "bumpy", "chequered", "cobwebbed", "cracked",
                          "crosshatched", "crystalline",
                          "dotted", "fibrous", "flecked", "freckled", "frilly", "gauzy", "grid", "grooved",
                          "honeycombed",
                          "interlaced", "knitted",
                          "lacelike", "lined", "marbled", "matted", "meshed", "paisley", "perforated", "pitted",
                          "pleated",
                          "polka-dotted", "porous",
                          "potholed", "scaly", "smeared", "spiralled", "sprinkled", "stained", "stratified", "striped",
                          "studded", "swirly", "veined",
                          "waffled", "woven", "wrinkled", "zigzagged"]

        IMAGE_NET_LABELS = \
            ["alp", "artichoke", "Band Aid", "bathing cap", "bookshop", "bull mastiff", "butcher shop",
             "carbonara", "chain", "chain saw", "chainlink fence", "cheetah", "cliff dwelling", "common iguana",
             "confectionery", "container ship", "corn", "crossword puzzle", "dishrag", "dock", "flat-coated retriever",
             "gibbon", "grocery store", "head cabbage", "honeycomb", "hook", "hourglass", "jigsaw puzzle",
             "jinrikisha", "lakeside", "lawn mower", "maillot", "microwave", "miniature poodle", "muzzle",
             "notebook", "ocarina", "orangutan", "organ", "paper towel", "partridge", "rapeseed",
             "sandbar", "sarong", "sea urchin", "shoe shop", "shower curtain", "stone wall", "theater curtain",
             "tile roof",
             "turnstile", "vault", "velvet", "window screen", "wool", "yellow lady's slipper"]

        IMAGE_NET_LABELS_S2 = \
            ["common iguana", "partridge", "flat-coated retriever", "bull mastiff", "miniature poodle", "cheetah",
             "sea urchin", "orangutan", "gibbon", "Band Aid", "bathing cap", "chain saw", "container ship", "hook",
             "hourglass", "jinrikisha", "lawn mower", "maillot", "microwave", "muzzle", "notebook", "ocarina", "organ",
             "paper towel", "sarong", "turnstile", "crossword puzzle", "yellow lady's slipper"
             ]

        IMAGE_NET_LABELS_T = \
            ["alp", "artichoke", "bookshop", "butcher shop",
             "carbonara", "chain", "chainlink fence", "cliff dwelling",
             "confectionery", "corn", "dishrag", "dock",
             "grocery store", "head cabbage", "honeycomb", "jigsaw puzzle",
             "lakeside", "rapeseed",
             "sandbar", "shoe shop", "shower curtain", "stone wall", "theater curtain", "tile roof",
             "vault", "velvet", "window screen", "wool", ]

        train_parameters = {
            "epochs": 400,
            "learning_rate": 0.0001,
            "texture_batch_size": 32,
            "image_net_batch_size": 32,
            "weight_decay": 0.0005
        }

        image_net_data_set_path = "./Dataset/ImageNet/ImageNet_X.pickle"
        image_net_label_set_path = "./Dataset/ImageNet/ImageNet_Y.pickle"
        image_net_test_path = "./Dataset/ImageNet/ImageNet_Test.pickle"

        image_net_S2_data_set_path = "./Dataset/ImageNet/ImageNet_S2X.pickle"
        image_net_S2_label_set_path = "./Dataset/ImageNet/ImageNet_S2Y.pickle"
        image_net_S2_test_path = "./Dataset/ImageNet/ImageNet_S2_Test.pickle"

        image_net_T_data_set_path = "./Dataset/ImageNet/ImageNet_TX.pickle"
        image_net_T_label_set_path = "./Dataset/ImageNet/ImageNet_TY.pickle"
        image_net_T_test_path = "./Dataset/ImageNet/ImageNet_T_Test.pickle"

        texture_train_data_set_path = "./Dataset/Texture/DTD/Texture_DTD_train{0}_X.pickle"
        texture_train_label_set_path = "./Dataset/Texture/DTD/Texture_DTD_train{0}_Y.pickle"

        texture_val_data_set_path = "./Dataset/Texture/DTD/Texture_DTD_val{0}_X.pickle"
        texture_val_label_set_path = "./Dataset/Texture/DTD/Texture_DTD_val{0}_Y.pickle"

        auto_encoder_model_path = "./Models/Auto_encoder_Model_epoch_300_lr_0.001_noise_factor_0.5.pt"
        saved_model_name = "./Models/MTL/DTD/Multitask_Classifier_Model_epoch_" + str(
            train_parameters["epochs"]) + "_lr_" + str(
            train_parameters["learning_rate"]) + "_split{0}.pth"

        split_size = 0.05

        # training starts
        texture_data_loader_list = DataPreProcessor.preprocess_DTD_train_val_10_splits(texture_train_data_set_path,
                                                                                       texture_train_label_set_path,
                                                                                       texture_val_data_set_path,
                                                                                       texture_val_label_set_path,
                                                                                       train_parameters[
                                                                                           "texture_batch_size"],
                                                                                       num_workers=0,
                                                                                       device=device)
        # image_net_S2_data_loader_dict = DataPreProcessor.preprocess_image_net(image_net_S2_data_set_path,
        #                                                                       image_net_S2_label_set_path,
        #                                                                       train_parameters[
        #                                                                           "image_net_batch_size"],
        #                                                                       image_net_S2_test_path,
        #                                                                       num_workers=0,
        #                                                                       split_size=split_size,
        #                                                                       device=device,
        #                                                                       type="ImageNet_S2"
        #                                                                       )
        # image_net_T_data_loader_dict = DataPreProcessor.preprocess_image_net(image_net_T_data_set_path,
        #                                                                      image_net_T_label_set_path,
        #                                                                      train_parameters[
        #                                                                          "image_net_batch_size"],
        #                                                                      image_net_T_test_path,
        #                                                                      num_workers=0,
        #                                                                      split_size=split_size,
        #                                                                      device=device,
        #                                                                      type="ImageNet_T"
        #                                                                      )

        image_net_data_loader_dict = DataPreProcessor.preprocess_image_net(image_net_data_set_path,
                                                                           image_net_label_set_path,
                                                                           train_parameters[
                                                                               "image_net_batch_size"],
                                                                           image_net_test_path,
                                                                           num_workers=0,
                                                                           split_size=split_size,
                                                                           device=device,
                                                                           type="ImageNet")
        train_arguments = {
            "IMAGE_NET_LABELS": IMAGE_NET_LABELS,
            "IMAGE_NET_LABELS_S2": IMAGE_NET_LABELS_S2,
            "IMAGE_NET_LABELS_T": IMAGE_NET_LABELS_T,
            "TEXTURE_LABELS": TEXTURE_LABELS,
            # "image_net_S2_data_loader_dict": image_net_S2_data_loader_dict,
            # "image_net_T_data_loader_dict": image_net_T_data_loader_dict,
            "image_net_data_loader_dict": image_net_data_loader_dict,
            "texture_data_loader_list": texture_data_loader_list,
            "train_parameters": train_parameters,
            "saved_model_name": saved_model_name
        }

        train = Train_Classifier()
        network = train.train_classifier(train_arguments, device, dataset_name="DTD")

        # training ends

        # testing starts
        texture_data_set_path = "./Dataset/Texture/DTD/Texture_DTD_test{0}_X.pickle"
        texture_label_set_path = "./Dataset/Texture/DTD/Texture_DTD_test{0}_Y.pickle"

        data_loader_test_list = DataPreProcessor.prepare_data_loader_test_10_splits(texture_data_set_path,
                                                                                    texture_label_set_path,
                                                                                    device)

        model_path_bn = "./Models/MTL/DTD/Multitask_Classifier_Model_epoch_400_lr_0.0001_split{0}.pth"

        test_arguments = {
            "data_loader_test_list": data_loader_test_list,
            "model_path_bn": model_path_bn,
            "TEXTURE_LABELS": TEXTURE_LABELS
        }
        # test(test_arguments, IMAGE_NET_LABELS, device)
        test = Test_Classifier()
        test.test_classifier(test_arguments, IMAGE_NET_LABELS, device, dataset_name="DTD")

    @staticmethod
    def surface_Train_test():
        # 0.05 test train val split
        device = Util.get_device()
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device)
        TEXTURE_LABELS = ['Kyberge_blanket1', 'Kyberge_canvas1', 'Kyberge_seat2', 'UIUC07_water',
                          'UIUC02_bark2', 'KTH_brown_bread', 'UIUC17_glass2',
                          'Kyberge_scarf1', 'KTH_corduroy', 'UIUC16_glass1', 'Kyberge_stoneslab1',
                          'Kyberge_rice2', 'UIUC06_wood3', 'KTH_aluminium_foil', 'Kyberge_ceiling1',
                          'Kyberge_sesameseeds1', 'Kyberge_floor2', 'Kyberge_lentils1', 'KTH_linen',
                          'UIUC08_granite', 'Kyberge_screen1', 'UIUC24_corduroy', 'Kyberge_oatmeal1',
                          'Kyberge_stone1', 'UIUC03_bark3', 'Kyberge_pearlsugar1', 'UIUC05_wood2',
                          'UIUC14_brick1', 'UIUC19_carpet2', 'UIUC23_knit', 'UIUC22_fur', 'UIUC15_brick2',
                          'KTH_wool', 'KTH_orange_peel', 'Kyberge_blanket2', 'Kyberge_sand1', 'KTH_sponge',
                          'Kyberge_seat1', 'Kyberge_scarf2', 'KTH_cracker', 'Kyberge_grass1', 'Kyberge_rice1',
                          'KTH_cork', 'UIUC04_wood1', 'Kyberge_cushion1', 'Kyberge_stone3', 'UIUC18_carpet1',
                          'Kyberge_ceiling2', 'UIUC10_floor1', 'Kyberge_floor1', 'Kyberge_stone2', 'KTH_cotton',
                          'UIUC09_marble', 'Kyberge_wall1', 'Kyberge_linseeds1', 'UIUC12_pebbles', 'UIUC11_floor2',
                          'UIUC01_bark1', 'Kyberge_rug1', 'KTH_styrofoam', 'UIUC25_plaid', 'UIUC21_wallpaper',
                          'UIUC13_wall',
                          'UIUC20_upholstery']

        IMAGE_NET_LABELS = \
            ["alp", "artichoke", "Band Aid", "bathing cap", "bookshop", "bull mastiff", "butcher shop",
             "carbonara", "chain", "chain saw", "chainlink fence", "cheetah", "cliff dwelling", "common iguana",
             "confectionery", "container ship", "corn", "crossword puzzle", "dishrag", "dock", "flat-coated retriever",
             "gibbon", "grocery store", "head cabbage", "honeycomb", "hook", "hourglass", "jigsaw puzzle",
             "jinrikisha", "lakeside", "lawn mower", "maillot", "microwave", "miniature poodle", "muzzle",
             "notebook", "ocarina", "orangutan", "organ", "paper towel", "partridge", "rapeseed",
             "sandbar", "sarong", "sea urchin", "shoe shop", "shower curtain", "stone wall", "theater curtain",
             "tile roof",
             "turnstile", "vault", "velvet", "window screen", "wool", "yellow lady's slipper"]

        train_parameters = {
            "epochs": 1000,
            "learning_rate": 0.0001,
            "texture_batch_size": 32,
            "image_net_batch_size": 32,
            "weight_decay": 0.0005
        }

        image_net_data_set_path = "./Dataset/ImageNet/ImageNet_X.pickle"
        image_net_label_set_path = "./Dataset/ImageNet/ImageNet_Y.pickle"
        image_net_test_path = "./Dataset/ImageNet/ImageNet_Test.pickle"

        texture_train_data_set_path = "./Dataset/Texture/Surface/Surface_X_train.pickle"
        texture_train_label_set_path = "./Dataset/Texture/Surface/Surface_Y_train.pickle"
        texture_test_data_set_path = "./Dataset/Texture/Surface/Surface_X_vaild.pickle"
        texture_test_label_set_path = "./Dataset/Texture/Surface/Surface_Y_vaild.pickle"

        saved_model_name = "./Models/MTL/Surface/Multitask_Classifier_Model_epoch_" + str(
            train_parameters["epochs"]) + "_lr_" + str(
            train_parameters["learning_rate"]) + "_split{0}.pth"

        split_size = 0.20

        # training starts
        texture_train_val_data_loader_list, texture_test_data_loader_list = \
            DataPreProcessor.preprocess_texture_surface(texture_train_data_set_path,
                                                        texture_train_label_set_path,
                                                        texture_test_data_set_path,
                                                        texture_test_label_set_path,
                                                        train_parameters["texture_batch_size"],
                                                        num_workers=0,
                                                        device=device, split_size=split_size,
                                                        type="Surface", folds=1)

        image_net_data_loader_dict = DataPreProcessor.preprocess_image_net(image_net_data_set_path,
                                                                           image_net_label_set_path,
                                                                           train_parameters[
                                                                               "image_net_batch_size"],
                                                                           image_net_test_path,
                                                                           num_workers=0,
                                                                           split_size=split_size,
                                                                           device=device,
                                                                           type="ImageNet")
        train_arguments = {
            "IMAGE_NET_LABELS": IMAGE_NET_LABELS,
            "TEXTURE_LABELS": TEXTURE_LABELS,
            "image_net_data_loader_dict": image_net_data_loader_dict,
            "texture_data_loader_list": texture_train_val_data_loader_list,
            "train_parameters": train_parameters,
            "saved_model_name": saved_model_name
        }

        train = Train_Classifier()
        network = train.train_classifier(train_arguments, device, dataset_name="kth")
        # training ends

        # testing starts

        model_path_bn = "./Models/MTL/Surface/Multitask_Classifier_Model_epoch_400_lr_0.0001_split{0}.pth"

        test_arguments = {
            "data_loader_test_list": texture_test_data_loader_list,
            "model_path_bn": model_path_bn,
            "TEXTURE_LABELS": TEXTURE_LABELS
        }

        test = Test_Classifier()
        accuracy_list, mean_accuracy = test.test_classifier(test_arguments, IMAGE_NET_LABELS, device)
        file1 = open("Surface_Details.txt", "a")
        file1.write(str(train_parameters))
        file1.write("Surface Mean accuracy: {0}\n".format(mean_accuracy))
        file1.write(str(accuracy_list))
        pd.DataFrame.from_dict(
            accuracy_list,
            orient='columns'
        ).to_csv("./Accuracy_Surface.csv")

    @staticmethod
    def kth_Train_test():
        # 0.05 test train val split
        device = Util.get_device()
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device)
        TEXTURE_LABELS = ["KTH_aluminium_foil", "KTH_brown_bread", "KTH_corduroy", "KTH_cork",
                          "KTH_cotton",
                          "KTH_cracker", "KTH_linen",
                          "KTH_orange_peel", "KTH_sponge", "KTH_styrofoam", "KTH_wool"]

        IMAGE_NET_LABELS = \
            ["alp", "artichoke", "Band Aid", "bathing cap", "bookshop", "bull mastiff", "butcher shop",
             "carbonara", "chain", "chain saw", "chainlink fence", "cheetah", "cliff dwelling", "common iguana",
             "confectionery", "container ship", "corn", "crossword puzzle", "dishrag", "dock", "flat-coated retriever",
             "gibbon", "grocery store", "head cabbage", "honeycomb", "hook", "hourglass", "jigsaw puzzle",
             "jinrikisha", "lakeside", "lawn mower", "maillot", "microwave", "miniature poodle", "muzzle",
             "notebook", "ocarina", "orangutan", "organ", "paper towel", "partridge", "rapeseed",
             "sandbar", "sarong", "sea urchin", "shoe shop", "shower curtain", "stone wall", "theater curtain",
             "tile roof",
             "turnstile", "vault", "velvet", "window screen", "wool", "yellow lady's slipper"]

        train_parameters = {
            "epochs": 400,
            "learning_rate": 0.0001,
            "texture_batch_size": 32,
            "image_net_batch_size": 32,
            "weight_decay": 0.0005
        }

        image_net_data_set_path = "./Dataset/ImageNet/ImageNet_X.pickle"
        image_net_label_set_path = "./Dataset/ImageNet/ImageNet_Y.pickle"
        image_net_test_path = "./Dataset/ImageNet/ImageNet_Test.pickle"

        texture_train_data_set_path = "./Dataset/Texture/kth/kth_X.pickle"
        texture_train_label_set_path = "./Dataset/Texture/kth/kth_Y.pickle"

        saved_model_name = "./Models/MTL/kth/Multitask_Classifier_Model_epoch_" + str(
            train_parameters["epochs"]) + "_lr_" + str(
            train_parameters["learning_rate"]) + "_split{0}.pth"

        split_size = 0.20

        # training starts
        texture_train_val_data_loader_list, texture_test_data_loader_list = \
            DataPreProcessor.preprocess_texture_except_DTD(texture_train_data_set_path,
                                                           texture_train_label_set_path,
                                                           train_parameters[
                                                               "texture_batch_size"],
                                                           num_workers=0,
                                                           device=device,
                                                           split_size=split_size,
                                                           type="Kth",
                                                           folds=1)

        image_net_data_loader_dict = DataPreProcessor.preprocess_image_net(image_net_data_set_path,
                                                                           image_net_label_set_path,
                                                                           train_parameters[
                                                                               "image_net_batch_size"],
                                                                           image_net_test_path,
                                                                           num_workers=0,
                                                                           split_size=split_size,
                                                                           device=device,
                                                                           type="ImageNet")
        train_arguments = {
            "IMAGE_NET_LABELS": IMAGE_NET_LABELS,
            "TEXTURE_LABELS": TEXTURE_LABELS,
            "image_net_data_loader_dict": image_net_data_loader_dict,
            "texture_data_loader_list": texture_train_val_data_loader_list,
            "train_parameters": train_parameters,
            "saved_model_name": saved_model_name
        }

        train = Train_Classifier()
        train.train_classifier(train_arguments, device, dataset_name="kth")
        # training ends

        # testing starts

        model_path_bn = "./Models/MTL/kth/Multitask_Classifier_Model_epoch_400_lr_0.0001_split{0}.pth"

        test_arguments = {
            "data_loader_test_list": texture_test_data_loader_list,
            "model_path_bn": model_path_bn,
            "TEXTURE_LABELS": TEXTURE_LABELS
        }

        test = Test_Classifier()
        accuracy_list, mean_accuracy = test.test_classifier(test_arguments, IMAGE_NET_LABELS, device)
        file1 = open("kth_Details.txt", "a")
        file1.write(str(train_parameters))
        file1.write("kth Mean accuracy: {0}\n".format(mean_accuracy))
        file1.write(str(accuracy_list))
        pd.DataFrame.from_dict(
            accuracy_list,
            orient='columns'
        ).to_csv("./Accuracy_Kth.csv")

    @staticmethod
    def kth_Train_test_single():
        # 0.05 test train val split
        device = Util.get_device()
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device)
        TEXTURE_LABELS = ["KTH_aluminium_foil", "KTH_brown_bread", "KTH_corduroy", "KTH_cork",
                          "KTH_cotton",
                          "KTH_cracker", "KTH_linen",
                          "KTH_orange_peel", "KTH_sponge", "KTH_styrofoam", "KTH_wool"]

        train_parameters = {
            "epochs": 25,
            "learning_rate": 0.0001,
            "texture_batch_size": 32,
            "image_net_batch_size": 32,
            "weight_decay": 0.0005
        }

        texture_train_data_set_path = "./Dataset/Texture/kth/kth_X.pickle"
        texture_train_label_set_path = "./Dataset/Texture/kth/kth_Y.pickle"

        saved_model_name = "./Models/MTL/kth/Multitask_Classifier_Model_epoch_" + str(
            train_parameters["epochs"]) + "_lr_" + str(
            train_parameters["learning_rate"]) + "_split{0}.pth"

        split_size = 0.20

        # training starts
        texture_train_val_data_loader_list, texture_test_data_loader_list = \
            DataPreProcessor.preprocess_texture_except_DTD(texture_train_data_set_path,
                                                           texture_train_label_set_path,
                                                           train_parameters[
                                                               "texture_batch_size"],
                                                           num_workers=0,
                                                           device=device,
                                                           split_size=split_size,
                                                           type="Kth",
                                                           folds=1)

        train_arguments = {
            "TEXTURE_LABELS": TEXTURE_LABELS,
            "texture_data_loader_list": texture_train_val_data_loader_list,
            "train_parameters": train_parameters,
            "saved_model_name": saved_model_name
        }

        train = Train_Classifier()
        train.train_classifier_single(train_arguments, device)
        # training ends

        # testing starts

        model_path_bn = "./Models/MTL/kth/Multitask_Classifier_Model_epoch_25_lr_0.0001_split{0}.pth"

        test_arguments = {
            "data_loader_test_list": texture_test_data_loader_list,
            "model_path_bn": model_path_bn,
            "TEXTURE_LABELS": TEXTURE_LABELS
        }

        test = Test_Classifier()
        accuracy_list, mean_accuracy = test.test_classifier_single(test_arguments, device)
        file1 = open("kth_Details.txt", "a")
        file1.write(str(train_parameters))
        file1.write("kth Mean accuracy: {0}\n".format(mean_accuracy))
        file1.write(str(accuracy_list))
        pd.DataFrame.from_dict(
            accuracy_list,
            orient='columns'
        ).to_csv("./Accuracy_Kth_single.csv")

    @staticmethod
    def fmd_Train_test():
        # 0.05 test train val split
        device = Util.get_device()
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device)
        TEXTURE_LABELS = ["fabric", "foliage", "glass", "leather", "metal", "paper", "plastic",
                          "stone", "water", "wood"]

        IMAGE_NET_LABELS = \
            ["alp", "artichoke", "Band Aid", "bathing cap", "bookshop", "bull mastiff", "butcher shop",
             "carbonara", "chain", "chain saw", "chainlink fence", "cheetah", "cliff dwelling", "common iguana",
             "confectionery", "container ship", "corn", "crossword puzzle", "dishrag", "dock", "flat-coated retriever",
             "gibbon", "grocery store", "head cabbage", "honeycomb", "hook", "hourglass", "jigsaw puzzle",
             "jinrikisha", "lakeside", "lawn mower", "maillot", "microwave", "miniature poodle", "muzzle",
             "notebook", "ocarina", "orangutan", "organ", "paper towel", "partridge", "rapeseed",
             "sandbar", "sarong", "sea urchin", "shoe shop", "shower curtain", "stone wall", "theater curtain",
             "tile roof",
             "turnstile", "vault", "velvet", "window screen", "wool", "yellow lady's slipper"]

        train_parameters = {
            "epochs": 2000,
            "learning_rate": 0.0001,
            "texture_batch_size": 32,
            "image_net_batch_size": 32,
            "weight_decay": 0.0005
        }

        image_net_data_set_path = "./Dataset/ImageNet/ImageNet_X.pickle"
        image_net_label_set_path = "./Dataset/ImageNet/ImageNet_Y.pickle"
        image_net_test_path = "./Dataset/ImageNet/ImageNet_Test.pickle"

        texture_train_data_set_path = "./Dataset/Texture/FMD/FMD_X.pickle"
        texture_train_label_set_path = "./Dataset/Texture/FMD/FMD_Y.pickle"

        saved_model_name = "./Models/MTL/FMD/Multitask_Classifier_Model_epoch_" + str(
            train_parameters["epochs"]) + "_lr_" + str(
            train_parameters["learning_rate"]) + "_split{0}.pth"

        split_size = 0.20

        # training starts
        texture_train_val_data_loader_list, texture_test_data_loader_list = \
            DataPreProcessor.preprocess_texture_except_DTD(texture_train_data_set_path,
                                                           texture_train_label_set_path,
                                                           train_parameters[
                                                               "texture_batch_size"],
                                                           num_workers=0,
                                                           device=device,
                                                           split_size=split_size,
                                                           type="FMD",
                                                           folds=1)

        image_net_data_loader_dict = DataPreProcessor.preprocess_image_net(image_net_data_set_path,
                                                                           image_net_label_set_path,
                                                                           train_parameters[
                                                                               "image_net_batch_size"],
                                                                           image_net_test_path,
                                                                           num_workers=0,
                                                                           split_size=split_size,
                                                                           device=device,
                                                                           type="ImageNet")
        train_arguments = {
            "IMAGE_NET_LABELS": IMAGE_NET_LABELS,
            "TEXTURE_LABELS": TEXTURE_LABELS,
            "image_net_data_loader_dict": image_net_data_loader_dict,
            "texture_data_loader_list": texture_train_val_data_loader_list,
            "train_parameters": train_parameters,
            "saved_model_name": saved_model_name
        }

        train = Train_Classifier()
        network = train.train_classifier(train_arguments, device, dataset_name="FMD")
        # training ends

        # testing starts

        model_path_bn = "./Models/MTL/kth/Multitask_Classifier_Model_epoch_120_lr_0.0001_split{0}.pth"

        test_arguments = {
            "data_loader_test_list": texture_test_data_loader_list,
            "model_path_bn": model_path_bn,
            "TEXTURE_LABELS": TEXTURE_LABELS
        }

        test = Test_Classifier()
        accuracy_list, mean_accuracy = test.test_classifier(test_arguments, IMAGE_NET_LABELS, device)
        file1 = open("FMD_Details.txt", "a")
        file1.write(str(train_parameters))
        file1.write("FMD  Mean accuracy: {0}\n".format(mean_accuracy))
        file1.write(str(accuracy_list))
        pd.DataFrame.from_dict(
            accuracy_list,
            orient='columns'
        ).to_csv("./Accuracy_FMD.csv")

    @staticmethod
    def uiuc_Train_test():
        device = Util.get_device()
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device)
        TEXTURE_LABELS = ["UIUC01_bark1", "UIUC02_bark2", "UIUC03_bark3", "UIUC04_wood1", "UIUC05_wood2",
                          "UIUC06_wood3",
                          "UIUC07_water", "UIUC08_granite", "UIUC09_marble", "UIUC10_floor1", "UIUC11_floor2",
                          "UIUC12_pebbles", "UIUC13_wall", "UIUC14_brick1", "UIUC15_brick2", "UIUC16_glass1",
                          "UIUC17_glass2",
                          "UIUC18_carpet1", "UIUC19_carpet2", "UIUC20_upholstery", "UIUC21_wallpaper", "UIUC22_fur",
                          "UIUC23_knit", "UIUC24_corduroy", "UIUC25_plaid"]

        IMAGE_NET_LABELS = \
            ["alp", "artichoke", "Band Aid", "bathing cap", "bookshop", "bull mastiff", "butcher shop",
             "carbonara", "chain", "chain saw", "chainlink fence", "cheetah", "cliff dwelling", "common iguana",
             "confectionery", "container ship", "corn", "crossword puzzle", "dishrag", "dock", "flat-coated retriever",
             "gibbon", "grocery store", "head cabbage", "honeycomb", "hook", "hourglass", "jigsaw puzzle",
             "jinrikisha", "lakeside", "lawn mower", "maillot", "microwave", "miniature poodle", "muzzle",
             "notebook", "ocarina", "orangutan", "organ", "paper towel", "partridge", "rapeseed",
             "sandbar", "sarong", "sea urchin", "shoe shop", "shower curtain", "stone wall", "theater curtain",
             "tile roof",
             "turnstile", "vault", "velvet", "window screen", "wool", "yellow lady's slipper"]

        train_parameters = {
            "epochs": 75,
            "learning_rate": 0.0001,
            "texture_batch_size": 32,
            "image_net_batch_size": 32,
            "weight_decay": 0.0005
        }

        image_net_data_set_path = "./Dataset/ImageNet/ImageNet_X.pickle"
        image_net_label_set_path = "./Dataset/ImageNet/ImageNet_Y.pickle"
        image_net_test_path = "./Dataset/ImageNet/ImageNet_Test.pickle"

        texture_train_data_set_path = "./Dataset/Texture/UIUC/UIUC_X.pickle"
        texture_train_label_set_path = "./Dataset/Texture/UIUC/UIUC_Y.pickle"

        saved_model_name = "./Models/MTL/UIUC/Multitask_Classifier_Model_epoch_" + str(
            train_parameters["epochs"]) + "_lr_" + str(
            train_parameters["learning_rate"]) + "_split{0}.pth"

        split_size = 0.20

        # training starts
        texture_train_val_data_loader_list, texture_test_data_loader_list = \
            DataPreProcessor.preprocess_texture_except_DTD(texture_train_data_set_path,
                                                           texture_train_label_set_path,
                                                           train_parameters[
                                                               "texture_batch_size"],
                                                           num_workers=0,
                                                           device=device,
                                                           split_size=split_size,
                                                           type="UIUC")

        image_net_data_loader_dict = DataPreProcessor.preprocess_image_net(image_net_data_set_path,
                                                                           image_net_label_set_path,
                                                                           train_parameters[
                                                                               "image_net_batch_size"],
                                                                           image_net_test_path,
                                                                           num_workers=0,
                                                                           split_size=split_size,
                                                                           device=device,
                                                                           type="ImageNet")
        train_arguments = {
            "IMAGE_NET_LABELS": IMAGE_NET_LABELS,
            "TEXTURE_LABELS": TEXTURE_LABELS,
            "image_net_data_loader_dict": image_net_data_loader_dict,
            "texture_data_loader_list": texture_train_val_data_loader_list,
            "train_parameters": train_parameters,
            "saved_model_name": saved_model_name
        }

        train = Train_Classifier()
        network = train.train_classifier(train_arguments, device, dataset_name="UIUC")
        # training ends

        # testing starts

        model_path_bn = "./Models/MTL/UIUC/Multitask_Classifier_Model_epoch_75_lr_0.0001_split{0}.pth"

        test_arguments = {
            "data_loader_test_list": texture_test_data_loader_list,
            "model_path_bn": model_path_bn,
            "TEXTURE_LABELS": TEXTURE_LABELS
        }

        test = Test_Classifier()
        accuracy_list, mean_accuracy = test.test_classifier(test_arguments, IMAGE_NET_LABELS, device,
                                                            dataset_name="UIUC")
        file1 = open("UIUC_Details.txt", "a")
        file1.write(str(train_parameters))
        file1.write("UIUC Mean accuracy: {0}\n".format(mean_accuracy))
        file1.write(str(accuracy_list))
        pd.DataFrame.from_dict(
            accuracy_list,
            orient='columns'
        ).to_csv("./Accuracy_UIUC.csv")

    @staticmethod
    def kylberg_Train_test():
        device = Util.get_device()
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device)
        TEXTURE_LABELS = ["Kyberge_blanket1", "Kyberge_blanket2", "Kyberge_canvas1",
                          "Kyberge_ceiling1", "Kyberge_ceiling2", "Kyberge_cushion1",
                          "Kyberge_floor1", "Kyberge_floor2", "Kyberge_grass1", "Kyberge_lentils1",
                          "Kyberge_linseeds1", "Kyberge_oatmeal1", "Kyberge_pearlsugar1", "Kyberge_rice1",
                          "Kyberge_rice2", "Kyberge_rug1", "Kyberge_sand1", "Kyberge_scarf1", "Kyberge_scarf2",
                          "Kyberge_sesameseeds1", "Kyberge_stone1", "Kyberge_stone2", "Kyberge_stone3",
                          "Kyberge_stoneslab1", "Kyberge_wall1"]

        IMAGE_NET_LABELS = \
            ["alp", "artichoke", "Band Aid", "bathing cap", "bookshop", "bull mastiff", "butcher shop",
             "carbonara", "chain", "chain saw", "chainlink fence", "cheetah", "cliff dwelling", "common iguana",
             "confectionery", "container ship", "corn", "crossword puzzle", "dishrag", "dock", "flat-coated retriever",
             "gibbon", "grocery store", "head cabbage", "honeycomb", "hook", "hourglass", "jigsaw puzzle",
             "jinrikisha", "lakeside", "lawn mower", "maillot", "microwave", "miniature poodle", "muzzle",
             "notebook", "ocarina", "orangutan", "organ", "paper towel", "partridge", "rapeseed",
             "sandbar", "sarong", "sea urchin", "shoe shop", "shower curtain", "stone wall", "theater curtain",
             "tile roof",
             "turnstile", "vault", "velvet", "window screen", "wool", "yellow lady's slipper"]

        train_parameters = {
            "epochs": 400,
            "learning_rate": 0.0001,
            "texture_batch_size": 32,
            "image_net_batch_size": 32,
            "weight_decay": 0.0005
        }

        image_net_data_set_path = "./Dataset/ImageNet/ImageNet_X.pickle"
        image_net_label_set_path = "./Dataset/ImageNet/ImageNet_Y.pickle"
        image_net_test_path = "./Dataset/ImageNet/ImageNet_Test.pickle"

        texture_train_data_set_path = "./Dataset/Texture/kylberg/kylbergs_X.pickle"
        texture_train_label_set_path = "./Dataset/Texture/kylberg/kylbergs_Y.pickle"

        saved_model_name = "./Models/MTL/kylberg/Multitask_Classifier_Model_epoch_" + str(
            train_parameters["epochs"]) + "_lr_" + str(
            train_parameters["learning_rate"]) + "_split{0}.pth"

        split_size = 0.20

        _accuracy_list = []

        # training starts

        texture_train_val_data_loader_list, texture_test_data_loader_list = \
            DataPreProcessor.preprocess_texture_except_DTD(texture_train_data_set_path,
                                                           texture_train_label_set_path,
                                                           train_parameters[
                                                               "texture_batch_size"],
                                                           num_workers=0,
                                                           device=device,
                                                           split_size=split_size,
                                                           type="kylberg",
                                                           folds=1)
        image_net_data_loader_dict = DataPreProcessor.preprocess_image_net(image_net_data_set_path,
                                                                           image_net_label_set_path,
                                                                           train_parameters[
                                                                               "image_net_batch_size"],
                                                                           image_net_test_path,
                                                                           num_workers=0,
                                                                           split_size=split_size,
                                                                           device=device,
                                                                           type="ImageNet")

        train_arguments = {
            "IMAGE_NET_LABELS": IMAGE_NET_LABELS,
            "TEXTURE_LABELS": TEXTURE_LABELS,
            "image_net_data_loader_dict": image_net_data_loader_dict,
            "texture_data_loader_list": texture_train_val_data_loader_list,
            "train_parameters": train_parameters,
            "saved_model_name": saved_model_name
        }

        train = Train_Classifier()
        network = train.train_classifier(train_arguments, device, dataset_name="kylberg")
        # training ends

        # testing starts

        model_path_bn = "./Models/MTL/kylberg/Multitask_Classifier_Model_epoch_400_lr_0.0001_split{0}.pth"

        test_arguments = {
            "data_loader_test_list": texture_test_data_loader_list,
            "model_path_bn": model_path_bn,
            "TEXTURE_LABELS": TEXTURE_LABELS
        }

        test = Test_Classifier()
        accuracy_list, mean_accuracy = test.test_classifier(test_arguments, IMAGE_NET_LABELS, device,
                                                            dataset_name="kylberg")

        file1 = open("kylberg_Details.txt", "a")
        file1.write(str(train_parameters))
        file1.write("kylberg Mean accuracy: {0}\n".format(mean_accuracy))
        file1.write(str(accuracy_list))
        pd.DataFrame.from_dict(
            accuracy_list,
            orient='columns'
        ).to_csv("./Accuracy_kylberg.csv")

    @staticmethod
    def curet_Train_test():
        device = Util.get_device()
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device)
        TEXTURE_LABELS = ["sample01", "sample02", "sample03", "sample04", "sample05", "sample05", "sample06",
                          "sample07",
                          "sample08", "sample09", "sample10", "sample11", "sample12", "sample13", "sample14",
                          "sample15",
                          "sample16", "sample17", "sample18", "sample19", "sample20", "sample21", "sample22",
                          "sample23",
                          "sample24", "sample25", "sample26", "sample27", "sample28", "sample29", "sample30",
                          "sample31",
                          "sample32", "sample33", "sample34", "sample35", "sample36", "sample37", "sample38",
                          "sample39",
                          "sample40", "sample41", "sample42", "sample43", "sample44", "sample45", "sample46",
                          "sample47",
                          "sample48", "sample49", "sample50", "sample51", "sample52", "sample53", "sample54",
                          "sample55",
                          "sample56", "sample57", "sample58", "sample59", "sample60", "sample61"]

        IMAGE_NET_LABELS = \
            ["alp", "artichoke", "Band Aid", "bathing cap", "bookshop", "bull mastiff", "butcher shop",
             "carbonara", "chain", "chain saw", "chainlink fence", "cheetah", "cliff dwelling", "common iguana",
             "confectionery", "container ship", "corn", "crossword puzzle", "dishrag", "dock", "flat-coated retriever",
             "gibbon", "grocery store", "head cabbage", "honeycomb", "hook", "hourglass", "jigsaw puzzle",
             "jinrikisha", "lakeside", "lawn mower", "maillot", "microwave", "miniature poodle", "muzzle",
             "notebook", "ocarina", "orangutan", "organ", "paper towel", "partridge", "rapeseed",
             "sandbar", "sarong", "sea urchin", "shoe shop", "shower curtain", "stone wall", "theater curtain",
             "tile roof",
             "turnstile", "vault", "velvet", "window screen", "wool", "yellow lady's slipper"]

        train_parameters = {
            "epochs": 75,
            "learning_rate": 0.0001,
            "texture_batch_size": 32,
            "image_net_batch_size": 32,
            "weight_decay": 0.0005
        }

        image_net_data_set_path = "./Dataset/ImageNet/ImageNet_X.pickle"
        image_net_label_set_path = "./Dataset/ImageNet/ImageNet_Y.pickle"
        image_net_test_path = "./Dataset/ImageNet/ImageNet_Test.pickle"

        texture_train_data_set_path = "./Dataset/Texture/CURET/CURET_X.pickle"
        texture_train_label_set_path = "./Dataset/Texture/CURET/CURET_Y.pickle"

        saved_model_name = "./Models/MTL/CURET/Multitask_Classifier_Model_epoch_" + str(
            train_parameters["epochs"]) + "_lr_" + str(
            train_parameters["learning_rate"]) + "_split{0}.pth"

        split_size = 0.20

        # training starts
        texture_train_val_data_loader_list, texture_test_data_loader_list = \
            DataPreProcessor.preprocess_texture_except_DTD(texture_train_data_set_path,
                                                           texture_train_label_set_path,
                                                           train_parameters[
                                                               "texture_batch_size"],
                                                           num_workers=0,
                                                           device=device,
                                                           split_size=split_size,
                                                           type="CURET")

        image_net_data_loader_dict = DataPreProcessor.preprocess_image_net(image_net_data_set_path,
                                                                           image_net_label_set_path,
                                                                           train_parameters[
                                                                               "image_net_batch_size"],
                                                                           image_net_test_path,
                                                                           num_workers=0,
                                                                           split_size=split_size,
                                                                           device=device,
                                                                           type="ImageNet")
        train_arguments = {
            "IMAGE_NET_LABELS": IMAGE_NET_LABELS,
            "TEXTURE_LABELS": TEXTURE_LABELS,
            "image_net_data_loader_dict": image_net_data_loader_dict,
            "texture_data_loader_list": texture_train_val_data_loader_list,
            "train_parameters": train_parameters,
            "saved_model_name": saved_model_name
        }

        train = Train_Classifier()
        network = train.train_classifier(train_arguments, device, dataset_name="CURET")
        # training ends

        # testing starts

        model_path_bn = "./Models/MTL/CURET/Multitask_Classifier_Model_epoch_75_lr_0.0001_split{0}.pth"

        test_arguments = {
            "data_loader_test_list": texture_test_data_loader_list,
            "model_path_bn": model_path_bn,
            "TEXTURE_LABELS": TEXTURE_LABELS
        }

        test = Test_Classifier()
        accuracy_list, mean_accuracy = test.test_classifier(test_arguments, IMAGE_NET_LABELS, device,
                                                            dataset_name="CURET")
        file1 = open("CURET_Details.txt", "a")
        file1.write(str(train_parameters))
        file1.write("CURET Mean accuracy: {0}\n".format(mean_accuracy))
        file1.write(str(accuracy_list))
        pd.DataFrame.from_dict(
            accuracy_list,
            orient='columns'
        ).to_csv("./Accuracy_CURET.csv")
