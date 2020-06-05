import numpy as np
import utils
from numpy import ndarray


class SPGD:
    def __init__(self, M: ndarray, k: int, momentum: float, sticky: bool):
        self.M = M
        self.k = k
        self.momentum = momentum
        self.sticky = sticky
        self.name = "Sticky GD" if sticky else "PGD"

    def grad(self, M: ndarray, W: ndarray, H: ndarray):
        gW = (W.dot(H) - M).dot(H.transpose())
        gH = W.transpose().dot(W.dot(H) - M)
        return (gW, gH)

    def optimize(self, step, iterations, save_image):
        (n, m) = self.M.shape
        M = self.M
        W = np.random.uniform(0, 1, (n, self.k))
        H = np.random.uniform(0, 1, (self.k, m))
        for i in range(iterations):
            print(i, utils.objective(M, W, H))
            (gW, gH) = self.grad(M, W, H)
            nW = np.random.normal(0, 0.1, (n, self.k))
            nH = np.random.normal(0, 0.1, (self.k, m))
            W = utils.proj(W - step * gW + nW, W, self.sticky)
            H = utils.proj(H - step * gH + nH, H, self.sticky)
        if save_image:
            utils.save_img(W.dot(H), 'spgd')
        print()
        return utils.objective(M, W, H)
