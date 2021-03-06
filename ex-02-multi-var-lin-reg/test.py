import pandas as pd

from knn_regressor import KNeighborsRegressor
from preprocessing import feature_normalization
from util import get_arg_or_else, hello, display_rmse

if __name__ == '__main__':
    test_path = get_arg_or_else(1, "data/test.csv")
    validation_set_percent = 1 - float(get_arg_or_else(2, 100))/100.0
    pd.options.mode.chained_assignment = None

    hello()

    # Read training data from file
    df = pd.read_csv('data/train.csv')

    # lower-case all column names
    df.columns = map(str.lower, df.columns)

    # Select specific features
    selected_features = ['sex', 'medu', 'reason', 'studytime', 'schoolsup', 'higher', 'goout', 'dalc', 'walc',
                         'internet', 'guardian', 'grade']
    binary_features = {
        'sex': ['F', 'M'],
        'schoolsup': ['yes', 'no'],
        'higher': ['yes', 'no'],
        'internet': ['yes', 'no']
    }
    nominal_features = {
        'reason': ['reputation', 'home', 'course', 'other'],
        'guardian': ['mother', 'father', 'other']
    }

    data = df[selected_features]

    # Min and max value from features should saved for test data
    norm_params = []
    # Normalization of features
    feature_normalization(data, binary=binary_features, nominal=nominal_features, norm_params=norm_params)

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

    # Create KNN model for data prediction
    knn = KNeighborsRegressor(k=19, distance=KNeighborsRegressor.euclidean_distance)
    # Set training data
    knn.fit(x_train.values.tolist(), y_train.values.tolist())

    if split:
        # Predict values for validation data
        y_validation_pred = knn.predict(x_validation.values.tolist())

        validation_rmse = KNeighborsRegressor.rmse(y_validation.values.tolist(), y_validation_pred)
        display_rmse(validation_rmse)

    # Read test data from file
    df_test = pd.read_csv(test_path)

    # lower-case all column names
    df_test.columns = map(str.lower, df_test.columns)

    test_data = df_test[selected_features]

    # Normalization of test features
    feature_normalization(test_data, binary=binary_features, nominal=nominal_features, norm_params=norm_params)

    x_test, y_true = test_data.ix[:, :-1], test_data.ix[:, -1]
    y_test_pred = knn.predict(x_test.values.tolist())

    # Calculate RMSE for test data
    test_rmse = KNeighborsRegressor.rmse(y_true.values.tolist(), y_test_pred)
    display_rmse(test_rmse, "test")
