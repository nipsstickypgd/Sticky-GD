import numpy as np
import torch
from torch import Tensor

import utils
from numpy import ndarray
import sklearn

from optimizers.Optimizer import Optimizer


class LeastSquares(Optimizer):
    # def __init__(self, M: ndarray, k: int):
    def __init__(self, M: Tensor, k: int):
        self.M = M
        self.k = k

    def name(self):
        return "HALS"

    def short_name(self):
        return "hals"

    def optimize(self, E, iterations):
        M = self.M
        W, H = utils.init_wh(M, self.k)
        for j in range(self.k):
            W[:, j] = W[:, j] / torch.norm(W[:, j])
            H[j, :] = H[j, :] / torch.norm(H[j, :])

        for i in range(iterations):
            print(i, utils.objective(M, W, H, E))
            Dif = E * (M - W.mm(H))
            for j in range(self.k):
                uj = W[:, j]
                vj = H[j, :]
                Dif = Dif + uj.reshape(-1, 1).mm(vj.reshape(1, -1))

                u_nom = Dif.t().mv(uj).clamp_min(0)
                if torch.norm(u_nom) > 1e-18:
                    vj = u_nom / (torch.norm(uj) ** 2)
                else:
                    vj = torch.zeros_like(vj)

                v_nom = Dif.mv(vj).clamp_min(0)
                if torch.norm(v_nom) > 1e-18:
                    uj = v_nom / (torch.norm(vj) ** 2)
                else:
                    uj = torch.zeros_like(uj)
                W[:, j] = uj
                H[j, :] = vj

                Dif = Dif - uj.reshape(-1, 1).mm(vj.reshape(1, -1))

        return W, H