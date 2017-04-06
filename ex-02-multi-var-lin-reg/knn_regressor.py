# coding=utf-8
import operator
from math import sqrt


class KNeighborsRegressor(object):
    def __init__(self, k, distance='L2'):
        """
        Constructor of KNeighborsRegressor

        :param k: number of neighbors to use
        :param distance: the distance metric to use for evaluation, possible values:
                    'L1': manhattan_distance
                    'L2': euclidean distance
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
        distance = 0
        for i in range(len(x)):
            distance += (x[i] - y[i]) ** 2

        return sqrt(distance)

    @staticmethod
    def manhattan_distance(x, y):
        """
        Function that calculate manhattan distance between two points

        :param x: input vector of features for one point
        :param y: input vector of features for one point
        :return: manhattan distance between two points
        """
        distance = 0
        for i in range(len(x)):
            distance += abs(x[i] - y[i])

        return distance

    def get_k_nearest_neighbour(self, unmarked_data):
        """
        Function that find k nearest neighbours for forwarded data

        :param unmarked_data: unmarked data
        :return: k nearest neighbours for unmarked_data
        """
        dists = {}

        for i in range(len(self.x)):
            if self.distance == "L1":
                d = self.manhattan_distance(self.x[i], unmarked_data)
            else:
                d = self.euclidean_distance(self.x[i], unmarked_data)
            dists[str(i)] = d

        # Sort all distances to get nearest points
        sorted_dists = sorted(dists.iteritems(), key=operator.itemgetter(1))

        k_neighbours = []
        for i in range(self.k):
            k_neighbours.append(self.y[int(sorted_dists[i][0])])

        return k_neighbours

    @staticmethod
    def vote(k_neighbours):
        """
        Function that find most voted class

        :param k_neighbours: k nearest neighbours
        :return: name of the class that has most occurrences
        """

        votes = {}

        for label in k_neighbours:
            if label in votes:
                votes[label] += 1
            else:
                votes[label] = 1

        sorted_votes = sorted(votes.iteritems(), key=operator.itemgetter(1), reverse=True)

        return sorted_votes[0][0]

    def predict(self, x):
        """
        Function that calculate predictions for data

        :param x: data used for prediction
        :return: predicted values for input data
        """
        results = []

        for i in range(len(x)):
            k_neighbours = self.get_k_nearest_neighbour(x[i])
            result = self.vote(k_neighbours)
            results.append(result)

        return results
