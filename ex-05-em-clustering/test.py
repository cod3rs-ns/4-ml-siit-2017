# -*- coding: utf-8 -*-

import pandas as pd

from exp_max import *
from preprocessing import read_image, preprocess_data
from util import hello, get_arg_or_else


def get_values_of_test_image(path):
    data = {
        'a_h': [],
        'b_s': [],
        'c_v': [],
        'd_path': []
    }
    read_image(path, data)
    return data


def find_cluster_of(probabilities):
    """
    Find cluster of probabilities.
    :param probabilities:  List of probabilities
    :return: name of the cluster
    """
    # select index of largest element
    ix = probabilities.index(max(probabilities))
    return {
        0: 'cloudy_sky',
        1: 'rivers',
        2: 'sunsets',
        3: 'trees_and_forests'
    }.get(ix)


epsilon = 0.01
clusters = 4
iters = 1000

if __name__ == "__main__":
    test_path = get_arg_or_else(1, "data/test.csv")
    data_process = get_arg_or_else(2, None)
    hello()

    if data_process:
        preprocess_data("data/images/")

    b = pd.read_csv("data/processed.csv")
    x = b.drop('d_path', axis=1)
    x = x.astype(float)
    # df_normalized = (x - x.min()) / (x.max() - x.min())
    df_normalized = x
    a = df_normalized.values

    # some average values for classes 'cloudy_sky', 'forest', 'sunset' and 'river in that particular order
    mu = [[0.557450406909, 0.570965982011, 0.631242031936], [0.297082985839, 0.536981188154, 0.644342852772, ],
          [0.0627345842156, 0.934576709652, 0.780241045252], [0.302649970548, 0.329837908527, 0.580027363411]]

    em = ExpectationMaximization(clusters, epsilon)
    results = em.fit(a, iters, mu)
    print("---------- results ----------")
    print("mu: " + str(results.mu))
    print("iters: " + str(results.num_iters))
    print("weights: " + str(["{0:.2f}%".format(x * 100) for x in results.w]))
    print("------ end of results -------\n")
    # show plot
    # gmm.plot_log_likelihood()

    # ========== Evaluation on test set ========== #
    test_data = pd.read_csv(test_path)
    test_x, test_y = test_data.ix[:, :-1], test_data.ix[:, -1]
    total_data = len(test_y)
    true_predicts = 0
    headers = True
    print("------------ test ------------")
    for x, y in zip(test_x.values.tolist(), test_y.values.tolist()):
        x = x[0]
        test_list = b.loc[b['d_path'] == x]
        t = [test_list.iloc[0]['a_h'], test_list.iloc[0]['b_s'], test_list.iloc[0]['c_v']]
        predicted_probabilities = em.predict(np.array(t)).tolist()
        predicted_cluster = find_cluster_of(predicted_probabilities)
        print("*-*-*-*-*-*-*-*-*-*")
        print("image: " + str(x))
        print("probabilities: " + str(predicted_probabilities))
        print("predicted: " + predicted_cluster)
        print("expected: " + str(y))
        print("*-*-*-*-*-*-*-*-*-*")
        if predicted_cluster == y:
            true_predicts += 1

    print "Accuracy for test set is {0:.2f}%.".format((100. * true_predicts) / total_data)

    """
    print "predictions ------------------------"
    print gmm.predict(np.array(df_normalized.values[40]))
    print b.iloc[40]['d_path']
    print (gmm.predict(np.array(df_normalized.values[0])))
    print b.iloc[0]['d_path']

    print 'zalazak'
    print (gmm.predict(np.array(df_normalized.values[140])))
    print b.iloc[140]['d_path']
    print (gmm.predict(np.array(df_normalized.values[161])))
    print b.iloc[161]['d_path']

    print (gmm.predict(np.array(df_normalized.values[581])))
    print b.iloc[581]['d_path']

    print "sume----------"
    print (gmm.predict(np.array(df_normalized.values[79])))
    print b.iloc[79]['d_path']
    print (gmm.predict(np.array(df_normalized.values[582])))
    print b.iloc[582]['d_path']
    print (gmm.predict(np.array(df_normalized.values[583])))
    print b.iloc[583]['d_path']
    print (gmm.predict(np.array(df_normalized.values[584])))
    print b.iloc[584]['d_path']
    print '-----------'

    print "nebo-------------"
    print (gmm.predict(np.array(df_normalized.values[71])))
    print b.iloc[71]['d_path']
    print (gmm.predict(np.array(df_normalized.values[72])))
    print b.iloc[72]['d_path']
    print (gmm.predict(np.array(df_normalized.values[119])))
    print b.iloc[119]['d_path']
    print "----------------------"
    """
