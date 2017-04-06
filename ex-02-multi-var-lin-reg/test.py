import pandas as pd
from sklearn.metrics import mean_squared_error

from knn_regressor import KNeighborsRegressor
from preprocessing import feature_normalization
from util import get_arg_or_else, hello

if __name__ == '__main__':
    test_path = get_arg_or_else(1, "data/test.csv")
    pd.options.mode.chained_assignment = None

    hello()

    # Read training data from file
    df = pd.read_csv('data/train.csv')

    # Select specific features
    selected_features = ['sex', 'Medu', 'reason', 'studytime', 'schoolsup', 'higher', 'goout', 'Dalc', 'Walc', 'Grade']
    data = df[selected_features]

    # Min and max value from features should saved for test data
    norm_params = []
    # Normalization of features
    feature_normalization(data, binary={'sex': ['F', 'M'], 'schoolsup': ['yes', 'no'], 'higher': ['yes', 'no']},
                          nominal={'reason': ['reputation', 'home', 'course', 'other']}, norm_params=norm_params)

    # Separate data to validation and train
    split_value = int(len(data) * 0.2)
    validation, train = data.ix[:split_value, :], data.ix[split_value:, :]

    x_train, y_train = train.ix[:, :-1], train.ix[:, -1]
    x_validation, y_validation = train.ix[:, :-1], train.ix[:, -1]

    # Create KNN model for data prediction
    knn = KNeighborsRegressor(k=2)
    # Set training data
    knn.fit(x_train.values.tolist(), y_train.values.tolist())
    # Predict values for validation data
    y_validation_pred = knn.predict(x_validation.values.tolist())

    validation_rmse = mean_squared_error(y_validation.values.tolist(), y_validation_pred) ** 0.5
    print "RMSE on validation data = {}".format(validation_rmse)

    # Read test data from file
    df_test = pd.read_csv(test_path)
    test_data = df[selected_features]

    # Normalization of test features
    feature_normalization(test_data, binary={'sex': ['F', 'M'], 'schoolsup': ['yes', 'no'], 'higher': ['yes', 'no']},
                          nominal={'reason': ['reputation', 'home', 'course', 'other']}, norm_params=norm_params)

    x_test, y_true = test_data.ix[:, :-1], test_data.ix[:, -1]
    y_test_pred = knn.predict(x_test.values.tolist())

    # Calculate RMSE for test data
    test_rmse = mean_squared_error(y_true.values.tolist(), y_test_pred) ** 0.5
    print "RMSE on test data = {}".format(test_rmse)
