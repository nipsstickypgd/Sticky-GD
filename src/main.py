import numpy as np

from optimizers.ao_admm import AO_ADMM
from optimizers.block_coordinate import BlockCoordinate
from optimizers.least_squares import LeastSquares
from optimizers.spgd import SPGD
from optimizers.mult_update import MultUpdate
from skimage.io import imread
import time

from PIL import Image

img = Image.open('data/kodak.png').convert('LA')
img.save('data/kodak_gray.png')


def read_image():
    img = imread('data/kodak_gray.png')
    return img[:, :, 0]


def rand_matrix():
    n = 1000
    m = 1000
    return np.random.uniform(0, 100, (n, m))


def main():
    k = 100
    iterations = 100

    isImg = True
    if isImg:
        step = 0.00002
        M = read_image()
        save_image = True
    else:
        M = rand_matrix()
        step = 0.00001
        save_image = False

    print(M.shape)
    names = []
    objs = []
    times = []
    # SPGD(M, k, 0, False),
    # BlockCoordinate(M, k, False)
    for opt in [SPGD(M, k, 0, True), MultUpdate(M, k), BlockCoordinate(M, k, False), AO_ADMM(M, k, 'kl'), LeastSquares(M, k)]:
        # for opt in [AO_ADMM(M, k, 'kl')]:
        print(opt.name)
        names.append(opt.name)
        start_time = time.time()
        objs.append(opt.optimize(step, iterations, save_image))
        elapsed = time.time() - start_time
        times.append(elapsed)
        print("time", elapsed)

    print(names)
    print(objs)
    print(times)


if __name__ == '__main__':
    main()
