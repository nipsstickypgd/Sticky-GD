import numpy as np
import utils
from numpy import ndarray


class MultUpdate:
    def __init__(self, M: ndarray, k: int):
        self.M = M
        self.k = k
        self.name = "Multiplicative update"

    def optimize(self, step, iterations, save_image):
        (n, m) = self.M.shape
        M = self.M
        W = np.random.uniform(1, 100, (n, self.k))
        H = np.random.uniform(1, 100, (self.k, m))
        for i in range(iterations):
            print(i, utils.objective(M, W, H))
            h_denom = W.transpose().dot(W).dot(H)
            H = H * np.where(h_denom < 1e-10, 1, W.transpose().dot(M) / h_denom)
            # print(np.min(W.dot(H).dot(H.transpose())))
            w_denom = W.dot(H).dot(H.transpose())
            W = W * np.where(w_denom < 1e-10, 1, M.dot(H.transpose()) / w_denom)
        if save_image:
            utils.save_img(W.dot(H), 'mult_update')
        print()
        return utils.objective(M, W, H)
