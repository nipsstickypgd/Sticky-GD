import numpy as np
from torch import Tensor

import utils
from numpy import ndarray

from optimizers.Optimizer import Optimizer


class BlockCoordinate(Optimizer):
    # def __init__(self, M: ndarray, k: int, step: float, sticky: bool):
    def __init__(self, M: Tensor, k: int, step: float, sticky: bool):
        self.M = M
        self.k = k
        self.step = step
        self.sticky = sticky

    def name(self):
        return "Sticky Block Coordinate" if self.sticky else "Block Coordinate"

    def short_name(self):
        return 'block_coordinate'

    def optimize(self, E, iterations):
        M = self.M
        W, H = utils.init_wh(M, self.k)
        for i in range(iterations):
            print(i, utils.objective(M, W, H, E))
            pH = H
            H = H + self.step * (W.T.mm(E * (M - W.mm(H))))
            H = utils.proj(H, pH, self.sticky)
            pW = W
            W = W + self.step * ((E * (M - W.mm(H))).mm(H.T))
            W = utils.proj(W, pW, self.sticky)

        return W, H
