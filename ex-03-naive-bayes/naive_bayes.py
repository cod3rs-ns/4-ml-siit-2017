import operator


class NaiveBayes(object):
    def __init__(self, train_inputs, train_outputs, hypotheses):
        """
        Constructor of Naive Bayes classifier
        
        :param train_inputs:    Matrix of train data
        :param train_outputs:   Vector (list) of target values
        :param hypotheses:      List of available classes - in the particular case [0, 1]
        """
        self._train_inputs = train_inputs
        self._train_outputs = train_outputs
        self._hypotheses = hypotheses
        # Number of features
        self._features_no = len(self._train_inputs[0])
        # Number of training examples
        self._examples_no = len(self._train_inputs)
        # No smoothing
        self._alpha = 0

    def with_smoothing(self, alpha):
        """
        Method that sets smoothing param to Naive Bayes
        
        :param alpha: smoothing param
        :return: 
        """
        self._alpha = alpha

    def predict(self, x, debug=False):
        """
        P(h|D) = P(D|h)*p(h)/p(D)
        We can ignore P(D) because it's same for all hypothesis.
        We have two hypothesis for this problem - 1 and 0.

        :param x: Input example we want to predict output value
        :param debug: If we want to print probabilities for each class then we should pass True
        """

        output = {}
        for h in self._hypotheses:
            #
            p_h = self._get_probability(h)
            # P(h|x1, x2, x3 ... xn) = P(h|x1) * P(h|x2) * ... * P(h|xn)
            p_hx = self._get_probability_of_hypothesis_if_data(x, h)

            if debug:
                print 'Probability of class {} for input {} is {}'.format(h, x, p_h * p_hx)

            output[h] = p_h * p_hx

        return sorted(output.items(), key=operator.itemgetter(1))[-1][0]

    def _get_probability(self, hypothesis):
        """
        Provides probabilistic of current hypothesis over the space
        
        :param hypothesis: concrete hypothesis we want to find probability for
        :return: probability of provided hypothesis
        """
        y = self._train_outputs
        return y.count(hypothesis) / (float(len(y)))

    def _get_probability_of_hypothesis_if_data(self, x, hypothesis):
        """
        Function that calculates probability of provided input
        
        :param x: input features
        :param hypothesis: concrete output value we are finding probability for
        :return: probability of 'hypothesis' for provided input 'x'
        """
        probability = 1
        # For each feature
        for feature in xrange(self._features_no):
            probability *= self._feature_probability(feature, x[feature], hypothesis)

        return probability

    def _feature_probability(self, feature, x, hypothesis, type="discrete"):
        """
        Function which calculates probability for each feature
        
        :param feature: index of concrete feature
        :param x: value of feature from provided input
        :param hypothesis: hypothesis we calculate probability for
        :param type: feature type ['discrete' or 'continuous']
        :return: probability of concrete input feature value
        """
        total_hypothesis_data = self._train_outputs.count(hypothesis)
        features_in_hypothesis = 0

        possible_feature_values = set()

        # Find probability that provided input feature is in provided training set for provided hypothesis
        for j in xrange(self._examples_no):
            possible_feature_values.add(self._train_inputs[j][feature])
            if self._train_outputs[j] == hypothesis and str(self._train_inputs[j][feature]) == str(x):
                features_in_hypothesis += 1

        possible_feature_values = len(possible_feature_values)

        return (float(features_in_hypothesis) + self._alpha) / \
               (total_hypothesis_data + self._alpha * possible_feature_values)
