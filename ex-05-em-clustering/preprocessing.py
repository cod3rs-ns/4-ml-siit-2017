# -*- coding: utf-8 -*-

from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
from skimage import io, color


def read_image(filename, data, path="data/images/"):
    """
    Convert images to usable data.
    :param filename: Image's file name
    :param data: Collection data object
    :param path: Folder path where images are stored
    :return: 
    """
    img = io.imread(path + filename)
    hsv_image = color.rgb2hsv(img)

    h, s, v = np.mean(hsv_image, axis=(0, 1))
    data['a_h'].append(h)
    data['b_s'].append(s)
    data['c_v'].append(v)

    data['d_path'].append(filename)


def preprocess_data(images_folder):
    """
    Process images from folder and save it to .csv file.
    :param images_folder: folder of where images are stored
    :return: 
    """
    only_files = [f for f in listdir(images_folder) if isfile(join(images_folder, f))]

    data = {
        'a_h': [],
        'b_s': [],
        'c_v': [],
        'd_path': []
    }

    for f in only_files:
        read_image(f, data, images_folder)

    df = pd.DataFrame.from_dict(data)
    df.to_csv("data/processed.csv", sep=',', encoding='utf-8', index=False)
