import numpy as np
import sklearn.model_selection as sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision.utils import save_image


# ------Model Start------------#
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # encoder layers
        self.enc1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.bn1 = nn.BatchNorm2d(96)
        self.enc2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.enc3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(3, 2)

        # decoder layers
        self.dec1 = nn.ConvTranspose2d(384, 256, kernel_size=3, stride=2)
        self.dec2 = nn.ConvTranspose2d(256, 96, kernel_size=3, stride=2)
        self.dec3 = nn.ConvTranspose2d(96, 3, kernel_size=11, stride=4)

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.float().cuda()
        else:
            x = x.float()
        # encode
        x = F.relu(self.bn1(self.enc1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.enc2(x)))
        x = self.pool(x)
        x = F.relu(self.enc3(x))  # the latent space representation

        # decode
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        # x = F.relu(self.dec4(x))
        # x = F.sigmoid(self.out(x))

        return x


# ------Model Start------------#

# ---------Util Start------------ #
def convert_to_tensor(X, Y, device):
    """
    Converts the dataset to tensor.

    :param X: dataset
    :param Y: label
    :param device: whether {cpu or gpu}

    :return: the dataset as tensor
    """
    tensor_x = torch.stack([torch.Tensor(i) for i in X])
    tensor_y = torch.from_numpy(Y)
    processed_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    return processed_dataset


# ---------Util End------------ #


# ---------DAL Start------------- #
def pre_process_data_set(data_path, label_path):
    """
    Preprocess the images in the data set to make each image dimension as 64 X 64.

    :param img_dims: image dimension

    :return: preprocessed the data set
    """

    data_set = np.load(data_path, allow_pickle=True)
    labels = np.load(label_path, allow_pickle=True)
    train_data = np.array(data_set)
    train_data = train_data / 255
    long_labels = np.array(labels, dtype=np.longlong)
    # print(long_labels)
    return train_data, long_labels


def read_dataset(data_set_path, label_set_path):
    """
    Reads the dataset.

    :param data_set_path:
    :param label_set_path:

    :return: dataset
    """
    train_data_set, labels_set = pre_process_data_set(data_set_path, label_set_path)
    return train_data_set, labels_set


def spilt_data_set(data_set, label_set, split_size):
    """
    Splits the data set into test and train set.

    :param data_set: dataset
    :param label_set: true labels
    :param split_size: split percentage

    :return: train and test dataset and corresponding labels
    """
    print(data_set.shape)
    print(label_set.shape)
    X_train, X_test, Y_train, Y_test = \
        sklearn.train_test_split(data_set, label_set, test_size=split_size, stratify=label_set)

    return X_train, X_test, Y_train, Y_test


# ---------DAL End------------- #

# ----------Train--------------#

def train(train_set, epochs, show_plot):
    print("Training")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network = Autoencoder().to(device)
    final_tot_correct = []
    batch_size = 32
    lr = 0.001
    noise_factor = 0.5
    # set batch size
    data_loader = torch.utils.data.DataLoader(train_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=64)

    # set optimizer - Adam
    optimizer = optim.Adam(network.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # start training
    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0

        for batch in data_loader:
            images, labels = batch
            # images = images.to(device)
            # labels = labels.to(device)

            noisy_imgs = images + noise_factor * torch.randn(*images.shape)
            noisy_imgs = np.clip(noisy_imgs, 0., 1.)

            images = images.to(device)
            labels = labels.to(device)
            noisy_imgs = noisy_imgs.to(device)

            # zero out grads for every new iteration
            optimizer.zero_grad()

            # forward propagation
            outputs = network(noisy_imgs)
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


def train_data_set(train_set,
                   epochs, show_plot):
    model = train(train_set,
                  epochs, show_plot);
    model_path_no_bn = "./imagenet_demo_100.pt"
    torch.save(model.state_dict(), model_path_no_bn)
    print('Saved model parameters to disk.')


def train_model(train_set,
                epochs, show_plot=False):
    model = train_data_set(train_set,
                           epochs, show_plot)
    return model


# ----------Train End----------#

# ---------BL Start------------- #
def split_train_test_validation_set(data_set_path, label_set_path, split_size, device):
    """
    This method splits the data set into train, test and validation set. Also this method resize the images
    based on image dimensions specified by image_dims parameter.

    :param data_set_path:
    :param label_set_path:
    :param split_size:
    :param device:
    :param flag:

    :return train, test and validation set and their corresponding sizes
    """
    train_data_set, labels_set = read_dataset(data_set_path, label_set_path)
    X_train, X_test, Y_train, Y_test = spilt_data_set(train_data_set,
                                                      labels_set,
                                                      split_size=split_size)
    train_set = convert_to_tensor(X_train, Y_train, device)
    test_set = convert_to_tensor(X_test, Y_test, device)

    return X_train, Y_train, train_set, test_set, Y_test.shape[0]


def final_test_train(train_set, test_set,
                     epochs,
                     test_set_size, classes):
    model_bn = train_model(train_set, epochs, show_plot=True)
    print("Training completed")


# ---------------BL End ------------- #

# --------------- Test Start----------- #
def test(test_data_set):
    print("Test set")
    model_path_bn = "./imagenet_demo_40.pt"
    data_loader = torch.utils.data.DataLoader(
        test_data_set, num_workers=1, shuffle=False, pin_memory=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    noise_factor = 0.5
    network = Autoencoder().to(device)
    network.load_state_dict(torch.load(model_path_bn, map_location=device))

    output = []
    idx = 1
    for batch in data_loader:
        img, _ = batch
        img_noisy = img + noise_factor * torch.randn(img.shape)
        img_noisy = np.clip(img_noisy, 0., 1.)

        img = img.to(device)
        img_noisy = img_noisy.to(device)

        outputs = network(img_noisy)
        # outputs = outputs.view(outputs.size(0), 1, 28, 28).cpu().data
        save_image(img_noisy, 'noisy_test_input.png')
        save_image(outputs, 'denoised_test_reconstruction.png')
        break


# --------------- Test End----------- #


# Unit testing
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_set_path = "./ImageNet_X.pickle"
label_set_path = "./ImageNet_Y.pickle"
split_size = 0.20

IMAGE_NET_LABELS = ["alp", "artichoke", "Band Aid", "bathing cap", "bookshop", "butcher shop", "carbonara", "chain",
                    "chainlink fence",
                    "cheetah", "cliff dwelling", "common iguana", "confectionery", "corn", "dishrag", "dock",
                    "flat-coated retriever",
                    "gibbon", "grocery store", "head cabbage", "honeycomb", "jigsaw puzzle", "lakeside",
                    "miniature poodle", "orangutan",
                    "partridge", "rapeseed", "sandbar", "sea urchin", "shoe shop", "shower curtain", "stone wall",
                    "theater curtain",
                    "tile roof", "vault", "velvet", "window screen", "wool"]

X_train, Y_train, train_set, test_set, test_set_size = \
    split_train_test_validation_set(data_set_path, label_set_path, split_size, device)
print(X_train.shape)

# net = Autoencoder()

final_test_train(train_set, test_set, 100, test_set_size, IMAGE_NET_LABELS)
# test(test_set)
# X = np.random.rand(2, 227, 227, 3)
# print(X)
# print(X.shape)
# X = X.swapaxes(1, 3)
# print(X)
# print(X.shape)
