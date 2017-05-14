from math import sqrt

import numpy as np
from numpy.linalg import det


def probability_in_cluster(data, mu, sigma):
    cluster_probability = np.ndarray(shape=(len(data),), dtype=float)
    for input_index in xrange(len(data)):
        cluster_probability[input_index] = probability_of_one_input(data[input_index], mu, sigma)

    return cluster_probability


def probability_of_one_input(x, mu, sigma):
    diff = x - mu
    nominator = np.exp(-0.5 * np.dot(diff, np.dot(np.linalg.inv(sigma), diff)))
    denominator = sqrt(((2 * np.pi) ** len(x)) * det(sigma))

    return nominator / denominator
