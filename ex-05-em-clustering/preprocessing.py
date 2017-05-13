from skimage import io, color
from matplotlib import pyplot as plt
import numpy as np


def read_image(filename, data, path="data\\images\\"):
    img = io.imread(path + filename)

    hsv_image = color.rgb2luv(img)

    #h_sep = img.shape[0] / 2
    #w_sep = img.shape[1] / 2

    #q1, q2, q3, q4 = hsv_image[:h_sep, :w_sep], hsv_image[:h_sep, w_sep:],  hsv_image[h_sep:, :w_sep], hsv_image[h_sep:, w_sep:]

    # print np.array(img).mean(axis=(0, 1))

    # io.imshow(img)
    # plt.show()
    r1, g1, b1 = np.mean(hsv_image, axis=(0, 1))
    data['r'].append(r1)
    data['g'].append(g1)
    data['b'].append(b1)

    #lab = color.rgb2lab(img)
    l, a, bb = np.min(hsv_image, axis=(0, 1))
    #data['l'].append(l)
    data['bb'].append((l + a)/2.0)

    '''
    r2, g2, b2 = np.mean(q2, axis=(0, 1))
    data['r2'].append(r2)
    data['g2'].append(g2)
    data['b2'].append(b2)

    r3, g3, b3 = np.mean(q3, axis=(0, 1))
    data['r3'].append(r3)
    data['g3'].append(g3)
    data['b3'].append(b3)

    r4, g4, b4 = np.mean(q4, axis=(0, 1))
    data['r4'].append(r4)
    data['g4'].append(g4)
    data['b4'].append(b4)
    '''
    data['path'].append(filename)
