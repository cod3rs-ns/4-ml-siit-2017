# -*- coding: utf-8 -*-

from probability import probability_in_cluster, probability_of_one_input
from copy import deepcopy

import numpy as np


class ExpectationMaximization:
    def __init__(self, clusters, eps=0.0001):
        self._clusters = clusters
        self._k = len(clusters)
        self._eps = eps
        self.output_params = None

    def fit(self, X, max_iters=1000, means=None):
        # n = number of input data, d = number of features for input data
        n, d = X.shape

        # randomize means
        if not means:
            random_mu = np.random.choice(n, self._k, False)
            mu = X[random_mu, :]
        else:
            mu = means

        # initialize the covariance matrices for each gaussian
        covariance = [np.eye(d)] * self._k

        # initialize the probabilities/weights for each gaussian
        w = [1.0 / self._k] * self._k

        # responsibility matrix is initialized to all zeros
        # we have responsibility for each of n points for each of clusters
        cluster_expectation = np.zeros((n, self._k))

        for iteration in xrange(max_iters):
            for ci in xrange(self._k):
                cluster_expectation[:, ci] = w[ci] * probability_in_cluster(X, mu[ci], covariance[ci])

            cluster_expectation = (cluster_expectation.T / np.sum(cluster_expectation, axis=1)).T

            n_ks = np.sum(cluster_expectation, axis=0)

            old_mu = deepcopy(mu)

            for ci in xrange(self._k):
                mu[ci] = 1.0 / n_ks[ci] * np.sum(cluster_expectation[:, ci] * X.T, axis=1).T
                x_mu = np.matrix(X - mu[ci])

                covariance[ci] = np.array(1 / n_ks[ci] * np.dot(np.multiply(x_mu.T, cluster_expectation[:, ci]), x_mu))

                w[ci] = 1. / n * n_ks[ci]

            # Check if algorithm has not significant changes
            if np.sum(np.absolute(np.array(mu) - np.array(old_mu))) < self._eps:
                break

        self.output_params = {"mu": mu, "cov": covariance, "w": w}
        return self.output_params

    def predict(self, data):
        clusters_mu = self.output_params["mu"]
        clusters_cov = self.output_params["cov"]
        clusters_w = self.output_params["w"]

        probabilities = []
        for cluster_mu, cluster_cov, cluster_w in zip(clusters_mu, clusters_cov, clusters_w):
            probabilities.append(cluster_w * probability_of_one_input(data, cluster_mu, cluster_cov))

        sum_of_probabilities = sum(probabilities)
        input_probabilities = map(lambda x: x / sum_of_probabilities, probabilities)

        cluster_index = input_probabilities.index(max(input_probabilities))
        return self._clusters[cluster_index]
