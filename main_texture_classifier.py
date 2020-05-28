from Util import Util
from datapreprocessor import DataPreProcessor
from train_classifier import Train_Classifier


def texture_classifier_BL():
    device = Util.get_device()
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    TEXTURE_LABELS = ["banded", "blotchy", "braided", "bubbly", "bumpy", "chequered", "cobwebbed", "cracked",
                      "crosshatched", "crystalline",
                      "dotted", "fibrous", "flecked", "freckled", "frilly", "gauzy", "grid", "grooved", "honeycombed",
                      "interlaced", "knitted",
                      "lacelike", "lined", "marbled", "matted", "meshed", "paisley", "perforated", "pitted", "pleated",
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
         "sandbar", "sarong", "sea urchin", "shoe shop", "shower curtain", "stone wall", "theater curtain", "tile roof",
         "turnstile", "vault", "velvet", "window screen", "wool", "yellow lady's slipper"]

    train_parameters = {
        "epochs": 400,
        "learning_rate": 0.0001,
        # "learning_rate": 0.0005,
        "texture_batch_size": 32,
        "image_net_batch_size": 32
    }

    image_net_data_set_path = "./Dataset/ImageNet/ImageNet_X.pickle"
    image_net_label_set_path = "./Dataset/ImageNet/ImageNet_Y.pickle"

    texture_train_data_set_path = "./Dataset/Texture/DTD/Texture_DTD_train{0}_X.pickle"
    texture_train_label_set_path = "./Dataset/Texture/DTD/Texture_DTD_train{0}_Y.pickle"

    texture_val_data_set_path = "./Dataset/Texture/DTD/Texture_DTD_val{0}_X.pickle"
    texture_val_label_set_path = "./Dataset/Texture/DTD/Texture_DTD_val{0}_Y.pickle"

    auto_encoder_model_path = "./Models/Auto_encoder_Model_epoch_300_lr_0.001_noise_factor_0.5.pt"
    saved_model_name = "./Models/Multitask_Classifier_Model_epoch_" + str(
        train_parameters["epochs"]) + "_lr_" + str(
        train_parameters["learning_rate"]) + "_split{0}.pth"

    dataset_labels = {
        "TEXTURE_LABELS": TEXTURE_LABELS,
        "IMAGE_NET_LABELS": IMAGE_NET_LABELS
    }

    split_size = 0.05

    texture_data_loader_list = DataPreProcessor.preprocess_DTD_train_val_10_splits(texture_train_data_set_path,
                                                                                   texture_train_label_set_path,
                                                                                   texture_val_data_set_path,
                                                                                   texture_val_label_set_path,
                                                                                   train_parameters[
                                                                                       "texture_batch_size"],
                                                                                   num_workers=0,
                                                                                   device=device)

    image_net_data_loader_dict = DataPreProcessor.preprocess_image_net(image_net_data_set_path,
                                                                       image_net_label_set_path,
                                                                       train_parameters[
                                                                           "image_net_batch_size"],
                                                                       num_workers=0,
                                                                       split_size=split_size,
                                                                       device=device)
    train_arguments = {
        "IMAGE_NET_LABELS": IMAGE_NET_LABELS,
        "TEXTURE_LABELS": TEXTURE_LABELS,
        "image_net_data_loader_dict": image_net_data_loader_dict,
        "texture_data_loader_list": texture_data_loader_list,
        "train_parameters": train_parameters,
        "saved_model_name": saved_model_name
    }

    train = Train_Classifier()
    network = train.train_classifier(train_arguments, device)

    task = "---validating"
    # test = validate_Auto_encoder()
    # test.validate_auto_encoder(val_set, model, task)


if __name__ == '__main__':
    texture_classifier_BL()
