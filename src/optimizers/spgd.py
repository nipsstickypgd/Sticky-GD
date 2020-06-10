from torch import Tensor

import utils

from optimizers.Optimizer import Optimizer


class SPGD(Optimizer):
    def __init__(self, M: Tensor, k: int, step: float, sticky: bool):
        # def __init__(self, M: ndarray, k: int, step: float, sticky: bool):
        self.M = M
        self.k = k
        self.step = step
        self.sticky = sticky

    def name(self):
        return "Sticky GD" if self.sticky else "PGD"

    def short_name(self):
        return "sticky_gd" if self.sticky else "pgd"

    def grad(self, M: Tensor, W: Tensor, H: Tensor, E: Tensor):
        # def grad(self, M: ndarray, W: ndarray, H: ndarray):
        gW = ((W.mm(H) - M) * E).mm(H.t())
        gH = W.t().mm((W.mm(H) - M) * E)
        # gW = (W.dot(H) - M).dot(H.transpose())
        # gH = W.transpose().dot(W.dot(H) - M)
        return gW, gH

    def optimize(self, E: Tensor, iterations):
        M = self.M
        W, H = utils.init_wh(M, self.k)
        for i in range(iterations):
            print(i, utils.objective(M, W, H, E))
            (gW, gH) = self.grad(M, W, H, E)
            # nW = np.random.normal(0, 0.01 * self.step, (n, self.k))
            # nH = np.random.normal(0, 0.01 * self.step, (self.k, m))
            W = utils.proj(W - self.step * gW, W, self.sticky)
            H = utils.proj(H - self.step * gH, H, self.sticky)

        return W, H
