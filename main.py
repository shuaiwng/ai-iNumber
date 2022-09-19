import numpy as np
import os
from PIL import Image


def read_img(path):
    img = Image.open(path).convert('L')
    pixels = img.load()
    arr = []
    for i in range(28):
        for j in range(28):
            arr.append(pixels[i,j])
    return arr


def get_files_list(data_path):
    files_list = []
    for path, subdirs, files in os.walk(data_path):
        for file in files:
            files_list.append(os.path.join(path,file))
    return files_list


if __name__ == "__main__":
    files_list = get_files_list("./mnist_png/training/")

    lth = len(files_list)
    for i in range(lth):
        read_img(files_list[i])
