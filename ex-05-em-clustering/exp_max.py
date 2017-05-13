# -*- coding: utf-8 -*-

from collections import namedtuple

import numpy as np
import pandas as pd
import pylab as plt


class ExpectationMaximization:
    def __init__(self, k=4, eps=0.0001):
        self.k = k
        self.eps = eps
        self.results = None

    def fit(self, X, max_iters=1000):

        # n = number of data-points, d = dimension of data points
        n, d = X.shape

        random_mu = np.random.choice(n, self.k, False)
        mu = X[random_mu, :]

        # initialize the covariance matrices for each gaussian
        covariance = [np.eye(d)] * self.k

        # initialize the probabilities/weights for each gaussian
        w = [1. / self.k] * self.k

        # responsibility matrix is initialized to all zeros
        # we have responsibility for each of n points for each of k gaussian
        responsibility = np.zeros((n, self.k))

        log_likelihoods = []

        probability = lambda mean, cov: np.linalg.det(cov) ** -.5 ** (2 * np.pi) ** (-X.shape[1] / 2.) * np.exp(
            -.5 * np.einsum('ij, ij -> i', X - mean, np.dot(np.linalg.inv(cov), (X - mean).T).T))

        # start algorithm
        while len(log_likelihoods) < max_iters:

            # ================================================ E step ================================================ #

            # vectorized implementation of e-step equation to calculate the membership for each of k-gaussian
            for ki in range(self.k):
                responsibility[:, ki] = w[ki] * probability(mu[ki], covariance[ki])

            # likelihood computation
            try:
                log_likelihood = np.sum(np.log(np.sum(responsibility, axis=1)))
            except Exception as e:
                continue

            log_likelihoods.append(log_likelihood)

            # normalize so that the responsibility matrix is row stochastic
            responsibility = (responsibility.T / np.sum(responsibility, axis=1)).T

            # the number of data-points belonging to each gaussian
            n_ks = np.sum(responsibility, axis=0)

            # ================================================ M step ================================================ #

            # calculate the new mean and covariance for each gaussian by utilizing the new responsibilities
            for ki in range(self.k):
                # means
                mu[ki] = 1. / n_ks[ki] * np.sum(responsibility[:, ki] * X.T, axis=1).T
                x_mu = np.matrix(X - mu[ki])

                # covariances
                covariance[ki] = np.array(1 / n_ks[ki] * np.dot(np.multiply(x_mu.T, responsibility[:, ki]), x_mu))

                # probabilities
                w[ki] = 1. / n * n_ks[ki]

            # check for convergence
            if len(log_likelihoods) < 2:
                continue
            if np.abs(log_likelihood - log_likelihoods[-2]) < self.eps:
                break

        # bind all results together
        self.results = namedtuple('results', ['mu', 'covariance', 'w', 'log_likelihoods', 'num_iters'])
        self.results.mu = mu
        self.results.covariance = covariance
        self.results.w = w
        self.results.num_iters = len(log_likelihoods)
        self.results.log_likelihoods = log_likelihoods

        return self.results

    def plot_log_likelihood(self):
        plt.plot(self.results.log_likelihoods)
        plt.title('Log Likelihood vs iteration plot')
        plt.xlabel('Iterations')
        plt.ylabel('log likelihood')
        plt.show()

    def predict(self, data):
        p = lambda mean, covariance: np.linalg.det(covariance) ** - 0.5 * (2 * np.pi) ** (-len(data) / 2) * np.exp(
            -0.5 * np.dot(data - mean, np.dot(np.linalg.inv(covariance),
                                              data - mean)))

        probabilities = np.array(
            [w * p(mu, s) for mu, s, w in zip(self.results.mu, self.results.covariance, self.results.w)])
        return probabilities / np.sum(probabilities)


if __name__ == "__main__":
    b = pd.read_csv("data/processed.csv")
    x = b.drop('path', axis=1)
    x = x.astype(float)
    df_normalized = (x - x.min()) / (x.max() - x.min())
    # print x.values[79]
    a = df_normalized.values

    epsilon = 0.01
    clusters = 4
    iters = 1000
    gmm = ExpectationMaximization(clusters, epsilon)
    results = gmm.fit(a, iters)
    # gmm.plot_log_likelihood()

    print "predictions ------------------------"
    print gmm.predict(np.array(df_normalized.values[40]))
    print b.iloc[40]['path']
    print (gmm.predict(np.array(df_normalized.values[0])))
    print b.iloc[0]['path']

    print 'zalazak'
    print (gmm.predict(np.array(df_normalized.values[140])))
    print b.iloc[140]['path']
    print (gmm.predict(np.array(df_normalized.values[161])))
    print b.iloc[161]['path']

    print (gmm.predict(np.array(df_normalized.values[581])))
    print b.iloc[581]['path']

    print "sume----------"
    print (gmm.predict(np.array(df_normalized.values[79])))
    print b.iloc[79]['path']
    print (gmm.predict(np.array(df_normalized.values[582])))
    print b.iloc[582]['path']
    print (gmm.predict(np.array(df_normalized.values[583])))
    print b.iloc[583]['path']
    print (gmm.predict(np.array(df_normalized.values[584])))
    print b.iloc[584]['path']
    print '-----------'

    print "nebo-------------"
    print (gmm.predict(np.array(df_normalized.values[71])))
    print b.iloc[71]['path']
    print (gmm.predict(np.array(df_normalized.values[72])))
    print b.iloc[72]['path']
    print (gmm.predict(np.array(df_normalized.values[119])))
    print b.iloc[119]['path']
    print "----------------------"
    # print "predictions ------------------------"
    # print("mu----------------------")
    # print (params.mu)
    # print("sigma----------------------")
    # print (params.Sigma)
    # print("iters----------------------")
    # print (params.num_iters)
    print("w----------------------")
    print (results.w)
