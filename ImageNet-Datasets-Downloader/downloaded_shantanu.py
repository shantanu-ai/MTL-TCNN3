import urllib.request

import pandas as pd


def url_to_jpg(i, url, file_path):
    filename = "image_{0}.jpg".format(i)
    fullPath = file_path + filename
    try:
        urllib.request.urlretrieve(url[0], fullPath)
        print("{0} save".format(filename))
    except Exception as e:
        pass


# set constants
FILENAME = "/Users/shantanughosh/Desktop/Shantanu_MS/Research/Dapeng_Wu/Git_Hub/Texture_Classification/ImageNet-Datasets-Downloader/Book2.csv"
FILE_PATH = "/Users/shantanughosh/Desktop/Shantanu_MS/Research/Dapeng_Wu/Git_Hub/Texture_Classification/ImageNet-Datasets-Downloader/data_root_folder/imagenet/"

urls = pd.read_csv(FILENAME)
idx = 0
for i, url in enumerate(urls.values):
    idx += 1
    url_to_jpg(i, url, FILE_PATH)

print("total: {0}".format(idx))
