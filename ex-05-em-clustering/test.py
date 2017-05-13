from preprocessing import read_image
import pandas as pd
from em import ExpectationMaximization

if __name__ == "__main__":
    '''
    # proccessing images and save to csv file

    from os import listdir
    from os.path import isfile, join

    folder = "data/images"
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]

    data = {
        'r': [],
        'g': [],
        'b': [],
        #'l': [],
        'bb': [],
        'path': []
    }

    for f in onlyfiles:
        read_image(f, data)

    df = pd.DataFrame.from_dict(data)
    df.to_csv("data/processed.csv", sep=',', encoding='utf-8', index=False)
    '''

    clusters = ['a', 'b', 'c', 'd']

    data = [
        [1, 2, 3],
        [1, 2, 3],
        [2, 3, 4]
    ]

    em = ExpectationMaximization(clusters)
    em.fit(data)
    em.predict()
