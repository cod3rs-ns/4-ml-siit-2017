# coding=utf-8
import operator
from math import sqrt
from functools import reduce
from sklearn.metrics import mean_squared_error


class KNeighborsRegressor(object):
    def __init__(self, k, distance):
        """
        Constructor of KNeighborsRegressor

        :param k: number of neighbors to use
        :param distance: function which we use for distance evaluating
        """
        self.k = k
        self.distance = distance
        self.x = []
        self.y = []

    def fit(self, x, y):
        """
        Fit the model using X as training data and y as target values

        :param x: input data for model prediction
        :param y: true values of output feature
        """
        self.x = x
        self.y = y

    @staticmethod
    def euclidean_distance(x, y):
        """
        Function that calculate euclidean distance between two points

        :param x: input vector of features for one point
        :param y: input vector of features for one point
        :return: euclidean distance between two points
        """
        return sqrt(reduce(operator.add, map(lambda a, b: (a - b) ** 2, x, y)))

    @staticmethod
    def manhattan_distance(x, y):
        """
        Function that calculate manhattan distance between two points

        :param x: input vector of features for one point
        :param y: input vector of features for one point
        :return: manhattan distance between two points
        """
        return reduce(operator.add, map(lambda a, b: abs(a - b), x, y))

    def get_k_nearest_neighbour(self, unmarked_data):
        """
        Function that find k nearest neighbours for forwarded data

        :param unmarked_data: unmarked data
        :return: k nearest neighbours for unmarked_data
        """
        dists = {}

        for i in xrange(len(self.x)):
            dists[str(i)] = self.distance(self.x[i], unmarked_data)

        # Sort all distances to get nearest points
        sorted_dists = sorted(dists.iteritems(), key=operator.itemgetter(1))

        k_neighbours = []
        for i in xrange(self.k):
            k_neighbours.append(self.y[int(sorted_dists[i][0])])

        return k_neighbours

    def vote(self, k_neighbours):
        """
        Function that find most voted class

        :param k_neighbours: k nearest neighbours
        :return: name of the class that has most occurrences
        """

        return sum(k_neighbours) / float(self.k)

    def predict(self, x):
        """
        Function that calculate predictions for data

        :param x: data used for prediction
        :return: predicted values for input data
        """
        results = []

        for i in xrange(len(x)):
            k_neighbours = self.get_k_nearest_neighbour(x[i])
            result = self.vote(k_neighbours)
            results.append(result)

        return results

    @staticmethod
    def rmse(y, y_predicted):
        return sqrt(mean_squared_error(y, y_predicted))

