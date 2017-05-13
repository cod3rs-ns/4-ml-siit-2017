from preprocessing import read_image
import pandas as pd

if __name__ == "__main__":
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

