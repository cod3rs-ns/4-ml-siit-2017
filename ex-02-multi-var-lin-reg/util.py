import sys


def get_arg_or_else(index, default=None):
    try:
        return sys.argv[index]
    except IndexError:
        return default


def hello():
    print "Tim 1:"
    print "SW 3/2013    Stefan Ristanovic"
    print "SW 9/2013    Bojan Blagojevic"
    print "SW F/2013    Dragutin Marjanovic"
    print
    print "You can run program with following arguments:"
    print "     python test.py file_path_to_test_data"
    print
    print
