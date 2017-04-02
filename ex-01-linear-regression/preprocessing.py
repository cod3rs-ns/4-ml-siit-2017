import numpy as np
import csv


def read_file(path="data/test.csv", skip_header=True):
    """
    Return 'x' and 'y' data from csv file with given path.
    Also, you can skip first row in file if you set 'skip_header' flag to 'True'.
    
    :param path: path to file, default is 'data/test.csv'
    :param skip_header: flag which tells to skip first row or not
    :return: tuple of float lists - X and Y data
    """

    x = []
    y = []
    with open(path, 'rb') as csv_file:
        reader = csv.reader(csv_file)

        # Skip header file
        if skip_header:
            next(reader)

        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))

    return x, y


def normalize(data):
    """
    Function which takes list of data and returns normalized data regarding to following function:
        x_normalized = (x - mean)/std
    where:
        mean    represents data mean value
        std     represents standard deviation
    
    :param data: list of data which should be normalized
    :return: normalized data
    """
    mean = np.median(data)
    std = np.std(data)

    return map(lambda val: (val - mean)/std, data)


def preprocess(data, method=(lambda val: np.log(val))):
    """
    Function which preprocess every element of data with provided method.
    Default method is log(x).
    :param data: list of data which should be preprocessed
    :param method: function we need to apply to each element of data
    :return: preprocessed data
    """
    return map(method, data)
