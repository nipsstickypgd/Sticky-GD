import utils
from common import device
from optimizers.Optimizer import Optimizer

import torch
from torch import Tensor


class AO_ADMM(Optimizer):
    def __init__(self, M: Tensor, k: int):
        self.M = M
        self.k = k

    def name(self):
        return "AO-ADMM"

    def short_name(self):
        return "ao_admm"

    def optimize(self, E, iterations):
        M = self.M
        W, H = utils.init_wh(M, self.k)
        V, U = utils.init_wh(M, self.k)

        for i in range(iterations):
            print(i, utils.objective(M, W, H, E))
            H, _ = admm_H_update(M, W, H, U, E, self.k)
            W, _ = admm_W_update(M, W, H, V, E, self.k)

        return W, H


def admm_H_update(M: Tensor, W: Tensor, H: Tensor, U: Tensor, E:Tensor, k, *, admm_iter=10):
    """
    ADMM update for NMF subproblem, when one of the factors is fixed using least-squares loss
    """

    g = W.T.mm(W)
    rho = torch.trace(g) / k
    L = torch.cholesky(g + rho * torch.eye(g.shape[0], device=device))
    WmH = W.mm(H)
    F = W.T.mm((M - WmH) * E + WmH)

    for i in range(admm_iter):
        H_aux = torch.cholesky_solve(F + rho * (H + U), L)
        H = (H_aux - U).clamp_min(0)
        U = U + H - H_aux

    return H, U


def admm_W_update(M: Tensor, W: Tensor, H: Tensor, V: Tensor, E:Tensor, k, *, admm_iter=10):
    """
    ADMM update for NMF subproblem, when one of the factors is fixed using least-squares loss
    """

    g = H.mm(H.T)
    rho = torch.trace(g) / k
    L = torch.cholesky(g + rho * torch.eye(g.shape[0], device=device))
    WmH = W.mm(H)
    F = ((M - WmH) * E + WmH).mm(H.T)

    for i in range(admm_iter):
        W_aux = torch.cholesky_solve((F + rho * (W + V)).T, L).T
        W = (W_aux - V).clamp_min(0)
        V = V + W - W_aux

    return W, V

# def prox(H_aux: Tensor, U: Tensor):
#     return (H_aux.T - U).clamp_min(0)
