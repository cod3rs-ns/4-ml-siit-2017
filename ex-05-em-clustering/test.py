from os import listdir
from os.path import isfile, join

import pandas as pd

from preprocessing import read_image

if __name__ == "__main__":

    # process images and save to csv file
    folder = "data/images"
    only_files = [f for f in listdir(folder) if isfile(join(folder, f))]

    data = {
        'a_h': [],
        'b_s': [],
        'c_v': [],
        'd_path': []
    }

    for f in only_files:
        read_image(f, data)

    df = pd.DataFrame.from_dict(data)
    df.to_csv("data/processed.csv", sep=',', encoding='utf-8', index=False)
