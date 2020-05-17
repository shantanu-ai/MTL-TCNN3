import os
import shutil

import numpy as np

# # Creating Train / Val / Test folders (One time use)
root_dir = "/Users/shantanughosh/Desktop/Shantanu_MS/Research/Dapen_Wu/data_root_folder/dataset/texture/dtd/"
classes_dir = ['/class1', 'class2', 'class3', 'class4']
TEXTURE_LABELS = ["banded", "blotchy", "braided", "bubbly", "bumpy", "chequered", "cobwebbed", "cracked",
                  "crosshatched", "crystalline",
                  "dotted", "fibrous", "flecked", "freckled", "frilly", "gauzy", "grid", "grooved", "honeycombed",
                  "interlaced", "knitted",
                  "lacelike", "lined", "marbled", "matted", "meshed", "paisley", "perforated", "pitted", "pleated",
                  "polka-dotted", "porous",
                  "potholed", "scaly", "smeared", "spiralled", "sprinkled", "stained", "stratified", "striped",
                  "studded", "swirly", "veined",
                  "waffled", "woven", "wrinkled", "zigzagged"]
# val_ratio = 0.15
val_ratio = 0
test_ratio = 0.10

for cls in TEXTURE_LABELS:
    os.makedirs(root_dir + '/test_images/' + cls)
    os.makedirs(root_dir + '/train_images/' + cls)
    # os.makedirs(root_dir + '/val_images' + cls)

    # Creating partitions of the data after shuffeling
    src = root_dir + "images/" + cls  # Folder to copy images from

    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)
    train_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                               [(int(len(allFileNames) * (1-test_ratio)))])

    train_FileNames = [src + '/' + name for name in train_FileNames.tolist()]
    # val_FileNames = [src + '/' + name for name in val_FileNames.tolist()]
    test_FileNames = [src + '/' + name for name in test_FileNames.tolist()]

    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    # print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))

    # Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name, root_dir + '/train_images/' + cls + "/")

    # for name in val_FileNames:
    #     shutil.copy(name, root_dir + '/val' + cls)

    for name in test_FileNames:
        shutil.copy(name, root_dir + '/test_images/' + cls + "/")
