from functools import partial

import pandas as pd
from sklearn import svm
from sklearn.metrics import f1_score

from util import get_arg_or_else, hello

# function that calculate mean value ignoring NaN values
nan_mean = partial(pd.DataFrame.mean, skipna=True)


def preprocess_data(data_frame, mean_values=None):
    """
    Function that prepare data for evaluation (interpolation of missing values and normalization)
    :param data_frame:  pandas data frame
    :param mean_values: dict of mean values for each column
    :return: normalized data frame without missing values
    """
    # replace missing values with NaN
    data = data_frame.replace('?', pd.np.nan)
    data = data.astype('float64')

    # interpolate missing values with mean value
    if mean_values is None:
        mean_values = nan_mean(data)

    data.fillna(mean_values, inplace=True)

    # normalize data
    df_normalized = (data - data.min()) / (data.max() - data.min())
    df_normalized['class'] = data['class']

    return df_normalized, mean_values


def get_test_data(path, mean_values):
    """
    Function that preprocess test data
    :param path: path to test data
    :param mean_values: mean_values to interpolate missing values
    :return: preprocessed test data
    """
    test_df = pd.read_csv(path)
    test_norm = preprocess_data(test_df, mean_values)[0]
    return test_norm.ix[:, :-1], test_norm.ix[:, -1]


if __name__ == "__main__":
    test_path = get_arg_or_else(1, "data/test.csv")
    validation_set_percent = 1 - float(get_arg_or_else(2, 100)) / 100.0
    hello()

    # Read training data from file
    df = pd.read_csv('data/train.csv')
    df_norm, means_for_train_data = preprocess_data(df)
    split = validation_set_percent > 0

    if split:
        # Separate data to validation and train
        split_value = int(len(df_norm) * validation_set_percent)
        train, validation = df_norm.ix[:split_value, :], df_norm.ix[split_value:, :]
    else:
        train = df_norm.ix[:, :]

    x_train, y_train = train.ix[:, :-1], train.ix[:, -1]

    # create svm model
    clf = svm.SVC(kernel='sigmoid')
    # clf = svm.LinearSVC()
    clf.fit(x_train, y_train)

    # ========== Evaluation on validation set ========== #
    if split:
        x_validation, y_validation = validation.ix[:, :-1], validation.ix[:, -1]
        y_pred = clf.predict(x_validation)
        micro_f1_score = f1_score(y_validation, y_pred, average='micro')
        print "Micro f1 score for validation set is {0:.7f}.".format(micro_f1_score)

    # ========== Evaluation on test set ========== #
    test_x, test_y = get_test_data(test_path, means_for_train_data)
    y_pred_test = clf.predict(test_x)
    micro_f1_score_test = f1_score(test_y, y_pred_test, average='micro')
    print "Micro f1 score for test set is {0:.7f}.".format(micro_f1_score_test)
