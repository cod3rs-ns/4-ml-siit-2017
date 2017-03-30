import numpy as np
import plotter

from preprocessing import read_file, preprocess, normalize
from lin_reg import linear_regression, rmse
from util import get_arg_or_else, hello

if __name__ == "__main__":
    path = get_arg_or_else(1, "data/test.csv")
    show_plot = get_arg_or_else(2, False)

    hello()

    # Read data from file
    X, Y = read_file(path)

    # Apply logarithmic function to each value
    Y_prep = preprocess(Y)

    # Save mean value and standard deviation for further use
    y_mean, y_std = np.mean(Y_prep), np.std(Y_prep)

    # Normalize input (X) and preprocessed output (Y)
    X_norm = normalize(X)
    Y_norm = normalize(preprocess(Y))

    # Return coefficients with normalized input and output
    slope, intercept = linear_regression(X_norm, Y_norm)

    # Predict outputs for training set
    Y_predicted = map(lambda x: np.exp((intercept + slope * x) * y_std + y_mean), X_norm)

    if show_plot:
        plotter.plot(X, Y, '.')
        plotter.plot(X, Y_predicted, 'ro')
        plotter.show()

    RMSE = rmse(X_norm, Y, lambda x: np.exp((intercept + slope * x) * y_std + y_mean))
    print "RMSE = {}".format(RMSE)