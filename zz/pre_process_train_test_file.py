import os
import shutil


def do_create_test_files(root_path, dst_dir_root, label_dir):
    # get the file names from source test folder
    contents = ""
    for i in range(10):
        file_name = root_path + label_dir + str(i + 1) + ".txt"
        with open(file_name) as f:
            for source_file in f.readlines():
                folder_name = source_file.split("/")[0]
                dst_dir = dst_dir_root + folder_name + "/"

                if not os.path.isdir(dst_dir):
                    os.mkdir(dst_dir)
                source_file_full_name = root_path + "images/" + source_file.rstrip("\n")
                print(dst_dir)
                print(source_file_full_name)

                # copy image
                shutil.copy(source_file_full_name, dst_dir)


if __name__ == '__main__':
    ROOT_PATH = "/Users/shantanughosh/Desktop/Shantanu_MS/Research/Dapen_Wu/data_root_folder/dataset/texture/dtd/"

    # test path
    test_dst_dir_root = ROOT_PATH + "test_images/"
    test_label_dir = "labels/test"

    # train path
    train_dst_dir_root = ROOT_PATH + "train_images/"
    train_label_dir = "labels/train"

    # do_create_test_files(ROOT_PATH, test_dst_dir_root, test_label_dir)
    do_create_test_files(ROOT_PATH, train_dst_dir_root, train_label_dir)
