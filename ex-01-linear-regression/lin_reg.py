import numpy as np


def linear_regression(x, y):
    """
    Function which evaluate 'slope' and 'intercept' coefficients for linear regression:
        y_hat = intercept + slope * x; 
    where:
        x       input value
        y_hat   predicted value
        
    :param x: input data
    :param y: output data
    :return: tuple of coefficients (slope, intercept)
    """
    assert (len(x) == len(y))

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = sum([xi * yi for xi, yi in zip(x, y)])
    denominator = sum([xi ** 2 for xi in x])

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    return slope, intercept


def rmse(x, y, y_hat):
    """
    Function which evaluate RMSE regarding to this formula:
        @link http://www.saedsayad.com/images/RMSE.png
    
    :param x: input values
    :param y: real output values
    :param y_hat: function we use for predicate output
    :return: evaluated Root Mean Square Error
    """

    assert (len(x) == len(y))
    n = len(x)

    return np.sqrt(sum(map(lambda (xi, yi): (yi - y_hat(xi))**2, zip(x, y)))/n)

