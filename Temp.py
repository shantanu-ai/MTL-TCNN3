import pickle

import matplotlib.pyplot as plt


def test_pickle():
    print("Test")
    pickle_in = open("ImageNet_X.pickle", "rb")
    X = pickle.load(pickle_in)
    print(X.shape)
    print(X[1].shape)
    X = X.swapaxes(1, 3)
    plt.imshow(X[1])
    plt.show()

    pickle_in = open("ImageNet_Y.pickle", "rb")
    Y = pickle.load(pickle_in)
    print(len(Y))
    print(Y[1])


# create_pickle_for_training()

test_pickle()
