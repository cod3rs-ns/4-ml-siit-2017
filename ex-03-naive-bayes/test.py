import pandas as pd

from naive_bayes import NaiveBayes

if __name__ == "__main__":
    # Read training data from file
    df = pd.read_csv('data/train.csv')

    # Select specific features
    selected_features = ['age', 'wedu', 'hedu', 'children', 'religion', 'employed', 'standard', 'media', 'contraceptive']

    data = df[selected_features]

    train = data.ix[:, :]
    x_train, y_train = train.ix[:, :-1], train.ix[:, -1]

    x = x_train.values.tolist()
    y = y_train.values.tolist()

    nb = NaiveBayes(x, y, [0, 1])

    print nb.predict([31, 3, 3, 3, 1, 0, 1, 0])
    print nb.predict([44, 4, 2, 8, 1, 0, 3, 1])
    print nb.predict([30, 3, 4, 4, 1, 0, 3, 1])
    print nb.predict([42, 4, 4, 4, 0, 1, 4, 1])
    print nb.predict([26, 4, 4, 2, 1, 1, 3, 1])
    print nb.predict([23, 3, 3, 2, 1, 1, 4, 1])
    print nb.predict([45, 1, 3, 10, 1, 0, 4, 1])
