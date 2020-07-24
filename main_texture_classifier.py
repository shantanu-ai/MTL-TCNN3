from Texture_Classification_Deep import Texture_Classification_Deep


def texture_classifier_BL():
    Texture_Classification_Deep.DTD_Train_test()
    Texture_Classification_Deep.kth_Train_test()
    # Texture_Classification_Deep.uiuc_Train_test()
    # Texture_Classification_Deep.kylberg_Train_test()
    # Texture_Classification_Deep.curet_Train_test()
    # Texture_Classification_Deep.fmd_Train_test()

    # Texture_Classification_Deep.kth_Train_test_single()


if __name__ == '__main__':
    texture_classifier_BL()
