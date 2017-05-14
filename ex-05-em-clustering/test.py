# -*- coding: utf-8 -*-

import pandas as pd

from exp_max import *
from preprocessing import read_image, preprocess_data
from util import hello, get_arg_or_else


def get_values_of_test_image(path):
    image_params = {
        'a_h': [],
        'b_s': [],
        'c_v': [],
        'd_path': []
    }
    read_image(path, image_params)
    return image_params


if __name__ == "__main__":
    test_path = get_arg_or_else(1, "data/test.csv")
    data_process = get_arg_or_else(2, None)
    hello()

    if data_process:
        preprocess_data("data/images/")

    input_features = pd.read_csv("data/processed.csv")
    data = input_features.drop('d_path', axis=1).astype(float)

    # Initial means for each cluster obtained from human's choice representative examples from data set
    mu = [[0.557450406909, 0.570965982011, 0.631242031936],   # cloudy_sky
          [0.297082985839, 0.536981188154, 0.644342852772],   # rivers
          [0.0627345842156, 0.934576709652, 0.780241045252],  # sunsets
          [0.302649970548, 0.329837908527, 0.580027363411]]   # trees_and_forests

    clusters = ['cloudy_sky', 'rivers', 'sunsets', 'trees_and_forests']

    em = ExpectationMaximization(clusters)
    results = em.fit(data.values, means=mu)

    # ========== Evaluation on test set ========== #
    test_data = pd.read_csv(test_path)
    file_paths, expected_clusters = test_data.ix[:, 0], test_data.ix[:, 1]

    total_data = len(file_paths)
    true_predicts = 0

    for file_path, expected_cluster in zip(file_paths.values, expected_clusters.values):
        test_list = get_values_of_test_image(file_path)
        t = [test_list['a_h'][0], test_list['b_s'][0], test_list['c_v'][0]]
        predicted_cluster = em.predict(np.array(t))

        if predicted_cluster == expected_cluster:
            true_predicts += 1

    print "Accuracy for test set is {0:.2f}%.".format((100. * true_predicts) / total_data)
