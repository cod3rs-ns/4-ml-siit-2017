def interval_category_splitter(x, intervals):
    """
    Function which determines to which category of interval belongs our real value
    
    :param x: real value
    :param intervals: boundary values of intervals we want to split
    :return: concrete interval value as string
    """
    if '?' == x:
        return '?'

    return str(len(filter(lambda _: int(x) > _, intervals)))


def make_intervals(data, intervals):
    """
    Function which transform vector of real values to discrete interval values
    
    :param data: vector of data we want transform
    :param intervals: boundary values of intervals we want to split
    :return: data with categorized intervals
    """
    return map(lambda x: interval_category_splitter(x, intervals), data)


def remove_non_values(data, char='?'):
    """
    Function which removes non-numeric values from data
    
    :param data: vector data we want to clean
    :param char: character/string which represents non-numeric value
    :return: clean data
    """
    return map(lambda _: int(_), filter(lambda _: _ != char, data))


def to_float(data):
    """
    Function which converts all list values to float
    
    :param data: clear data with mixed values (strings)
    :return: list of floats
    """
    return map(lambda _: float(_), remove_non_values(data))