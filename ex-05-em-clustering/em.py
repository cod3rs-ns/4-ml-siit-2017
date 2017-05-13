from random import random
from scipy.stats import norm, multivariate_normal


class ExpectationMaximization(object):
    def __init__(self, classes):
        """
        Constructor for Expectation Maximization algorithm
        
        :param classes: list of clusters
        """
        self._classes = classes
        self._k = len(classes)
        self._data = None

    def fit(self, data):
        """
        Method that evaluates the best hypothesis 
        
        :param data: List of training examples
        :return: best fitting hypothesis for clustering unknown examples
        """
        for row in data:
            row += self._k * [None]

        self._data = data

    def predict(self):
        """
        Method that returns predicted cluster for provided input
        
        # :param x: training input
        :return: predicted class of cluster
        """

        values = [[self._means(), self._variances()] for i in xrange(self._k)]

        epsilon = 0.0
        # print  values

        for iteration in xrange(9):
            for row in self._data:
                row[-self._k:] = map(
                    lambda (index, el):
                        ExpectationMaximization._probability_cluster_input(index, row[:-self._k], values),
                    enumerate(row[-self._k:])
                )

            estimated_values = values
            # if abs(estimated_values - values) < epsilon:
            # if True:
            #     break
            # else:
            #     values = estimated_values

        # print self._data

    @staticmethod
    def _probability_cluster_input(ci, X, values):
        """
        Method that returns estimated probability that input X belongs to cluster with index 'ci'
        :param ci: cluster index
        :param X: input parameters
        :param values: list of parameters estimated cluster distributions
        :return: 
        """
        prob = 1
        suma = 0
        # print X
        # for xi in X:
        var = multivariate_normal(mean=values[ci][0], cov=values[ci][1])
        prob *= var.pdf(X)
        # print prob
        # norm.pdf(X, values[ci][0], values[ci][1])

        for i in xrange(len(values)):
            a = 0.25
            # for xi in X:
            var = multivariate_normal(mean=values[i][0], cov=values[i][1])
            a *= var.pdf(X)
            suma += a

        print prob/(suma**2)
        return prob/(suma**2)


    def _means(self):
        return [random() for i in xrange(3)]

    def _variances(self):
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
