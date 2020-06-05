import numpy as np
import utils
from numpy import ndarray


class BlockCoordinate:
    def __init__(self, M: ndarray, k: int, sticky: bool):
        self.M = M
        self.k = k
        self.sticky = sticky
        self.name = "Sticky Block Coordinate" if sticky else "Block Coordinate"

    def optimize(self, step, iterations, save_image):
        step /= 3
        (n, m) = self.M.shape
        M = self.M
        W = np.random.uniform(0, 1, (n, self.k))
        H = np.random.uniform(0, 1, (self.k, m))
        for i in range(iterations):
            print(i, utils.objective(M, W, H))
            pH = H
            H = H + step * (W.transpose().dot(M) - W.transpose().dot(W).dot(H))
            H = utils.proj(H, pH, self.sticky)
            pW = W
            W = W + step * (M.dot(H.transpose()) - W.dot(H).dot(H.transpose()))
            W = utils.proj(W, pW, self.sticky)
        if save_image:
            utils.save_img(W.dot(H), 'block_coordinate')
        print()
        return utils.objective(M, W, H)
