import matplotlib.pyplot as plt


def predict(x, slope, intercept):
    return intercept + slope * x


def create_line(x, slope, intercept):
    y = [predict(xx, slope, intercept) for xx in x]
    return y


def plot(x, y, attr):
    plt.plot(x, y, attr)


def show():
    plt.show()
