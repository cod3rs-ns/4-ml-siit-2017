class ExpectationMaximization(object):
    def __init__(self, classes):
        """
        Constructor for Expectation Maximization algorithm
        
        :param classes: list of clusters
        """
        self._classes = classes
        self._k = len(classes)

    def fit(self, data):
        """
        Method that evaluates the best hypothesis 
        
        :param data: List of training examples
        :return: best fitting hypothesis for clustering unknown examples
        """
        pass

    def predict(self, x):
        """
        Method that returns predicted cluster for provided input
        
        :param x: training input
        :return: predicted class of cluster
        """
        pass
