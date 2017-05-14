import numpy as np
from skimage import io, color


def read_image(filename, data, path="data/images/"):
    img = io.imread(path + filename)
    hsv_image = color.rgb2hsv(img)

    h, s, v = np.mean(hsv_image, axis=(0, 1))
    data['a_h'].append(h)
    data['b_s'].append(s)
    data['c_v'].append(v)

    data['d_path'].append(filename)
