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
        y = self._train_outputs
        return y.count(hypothesis) / (float(len(y)))

    def _get_probability_of_hypothesis_if_data(self, x, hypothesis):
        probability = 1
        n = self._train_outputs.count(hypothesis)
        # For each attribute
        for i in xrange(len(self._train_inputs[0])):
            s = 0
            for j in xrange(len(self._train_inputs)):
                if self._train_outputs[j] == hypothesis and str(self._train_inputs[j][i]) == str(x[i]):
                    s += 1

            probability *= float(s) / n

        return probability
