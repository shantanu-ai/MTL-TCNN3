def test_pickle():
    a = [[
        [1, 2, 3], 1],
        [[4, 5, 6], 2]
    ]

    b = [[
        [11, 21, 31], 1],
        [[42, 52, 62], 2]
    ]
    x = []
    y = ["task1", "task2"]
    for i in a:
        x.append({y[0]: i})

    for i in b:
        x.append({y[1]: i})

    for i in x:
        for key, value in i.items() :
            print(key)
            print(value)
        # print(i)

    print(len(x))

    # print("Test")
    # pickle_in = open("ImageNet_X.pickle", "rb")
    # X = pickle.load(pickle_in)
    # print(X.shape)
    # print(X[1].shape)
    # X = X.swapaxes(1, 3)
    # plt.imshow(X[1])
    # plt.show()
    #
    # pickle_in = open("ImageNet_Y.pickle", "rb")
    # Y = pickle.load(pickle_in)
    # print(len(Y))
    # print(Y[1])


# create_pickle_for_training()

test_pickle()
