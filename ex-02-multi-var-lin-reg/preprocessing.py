import numpy as np


def feature_normalization(data, binary, nominal, range_cols=None, norm_params=[], feature_to_predict='grade'):
    """
    Function which takes list of data and normalize numeric data and
    convert nominal and binary data to appropriate numerical representation

    :param data: data to be processed
    :param binary: dict in the form of dict['feature_name']=['bin_value_1', 'bin_value_2']
    :param nominal: dict in the form of dict['feature_name']=['category_1', 'category_2', ...]
    :param range_cols: optional list of feature names that are in numerical range
    :param norm_params: params for normalization of data whose representation is range of values
    :param feature_to_predict: name of feature that should be predicted
    """
    if range_cols is None:
        range_cols = [col for col in data.columns.values
                      if col not in binary.keys() + nominal.keys() + [feature_to_predict]]

    rp_initialized = len(norm_params) != 0

    # Normalization of features whose values are in some range
    for i in xrange(len(range_cols)):
        col_name = range_cols[i]
        col_values = data[col_name]

        if not rp_initialized:
            median = np.median(col_values)
            std = np.std(col_values)
            norm_params.append((median, std))
        else:
            median = norm_params[i][0]
            std = norm_params[i][1]

        data[col_name] = (col_values - median) / std

    # Set binary feature values from string to 0 or 1
    for col_name, bin_values in binary.items():
        data[col_name] = data[col_name].map({bin_values[0]: 1, bin_values[1]: 0})

    # Create new feature for every category from nominal feature
    for col_name in nominal:
        for cat in nominal[col_name]:
            data[cat] = map(lambda x: int(x == cat), data[col_name])

        data.drop(col_name, axis=1, inplace=True)

    # Set column to be predicted at the last position
    y_values = data[feature_to_predict]
    data.drop(feature_to_predict, axis=1, inplace=True)
    data[feature_to_predict] = y_values
