import pandas as pd

from naive_bayes import NaiveBayes
from preprocessing import make_intervals
from util import get_arg_or_else, hello

# Specify features we want to read from file
selected_features = ['age', 'wedu', 'hedu', 'children', 'religion', 'employed', 'standard', 'media', 'contraceptive']


def get_test_data(path):
    """
    Function which read data from file and return separate inputs and targeted outputs
    
    :param path: file path
    :return: tuple of input matrix and targeted output 
    """
    test_df = pd.read_csv(path)

    test = test_df[selected_features]
    test_data = test.ix[:, :]
    x_test, y_test = test_data.ix[:, :-1], test_data.ix[:, -1]

    return x_test.values.tolist(), y_test.values.tolist()


if __name__ == "__main__":
    test_path = get_arg_or_else(1, "data/test.csv")
    validation_set_percent = 1 - float(get_arg_or_else(2, 20)) / 100.0
    create_intervals = get_arg_or_else(3, False)

    hello()

    # Read training data from file
    df = pd.read_csv('data/train.csv')
    data = df[selected_features]

    split = validation_set_percent > 0

    if split:
        # Separate data to validation and train
        split_value = int(len(data) * validation_set_percent)
        train, validation = data.ix[:split_value, :], data.ix[split_value:, :]
    else:
        train = data.ix[:, :]

    x_train, y_train = train.ix[:, :-1], train.ix[:, -1]

    if split:
        x_validation, y_validation = validation.ix[:, :-1], validation.ix[:, -1]

    if create_intervals:
        # TODO You must to specify your intervals
        age_intervals = [0, 10, 20, 30, 40]
        children_intervals = [0, 2, 4, 6]

        x_train['age'] = make_intervals(x_train['age'], age_intervals)
        x_train['children'] = make_intervals(x_train['children'], children_intervals)

        x_validation['age'] = make_intervals(x_validation['age'], age_intervals)
        x_validation['children'] = make_intervals(x_validation['children'], age_intervals)

    # Initialize Naive Bayes
    x = x_train.values.tolist()
    y = y_train.values.tolist()
    nb = NaiveBayes(x, y, [0, 1])
    nb.with_smoothing(1)

    if split:
        # Evaluate on validation set
        total_data = len(y_validation.values.tolist())
        true_predicts = 0
        for x, y in zip(x_validation.values.tolist(), y_validation.values.tolist()):
            predicted_value = nb.predict(x)
            if predicted_value == y:
                true_predicts += 1

        print "Accuracy for validation set is {0:.2f}%.".format((100. * true_predicts) / total_data)

    # Evaluate on testing set
    test_x, test_y = get_test_data(test_path)

    total_data = len(test_y)
    true_predicts = 0
    for x, y in zip(test_x, test_y):
        predicted_value = nb.predict(x)
        if predicted_value == y:
            true_predicts += 1

    print "Accuracy for test set is {0:.2f}%.".format((100.*true_predicts)/total_data)
