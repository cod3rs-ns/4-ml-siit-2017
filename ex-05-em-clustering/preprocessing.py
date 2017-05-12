from skimage import io


def read_image(filename, path="data/images/"):
    # io.use_plugin('qt')
    im = io.imread(path + filename)
    print(im)
    io.imshow(im)
    # io.show()

