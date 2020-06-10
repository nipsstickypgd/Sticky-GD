import torch
from torch import Tensor

import utils

from common import eps
from optimizers.Optimizer import Optimizer


class MultUpdate(Optimizer):
    # def __init__(self, M: ndarray, k: int):
    def __init__(self, M: Tensor, k: int):
        self.M = M
        self.k = k

    def name(self):
        return "Multiplicative update"

    def short_name(self):
        return "mu"

    def optimize(self, E, iterations):
        M = self.M
        W, H = utils.init_wh(M, self.k)
        for i in range(iterations):
            print(i, utils.objective(M, W, H, E))
            # M_aux = (M - W.mm(H)) * E + W.mm(H)
            h_denom = W.t().mm(E * W.mm(H))
            H = H * torch.where(h_denom < 1e-10, eps, W.t().mm(M) / h_denom) # TODO
            # M_aux = (M - W.mm(H)) * E + W.mm(H)
            w_denom = (E * W.mm(H)).mm(H.t())
            W = W * torch.where(w_denom < 1e-10, eps, (M).mm(H.t()) / w_denom) # TODO

        return W, H
