from math import sqrt
from typing import Tuple

from common import device
import torch
from numpy import ndarray
from skimage.io import imsave
from torch import Tensor


def frob_norm(M: Tensor) -> float:
    return torch.norm(M, 'fro').item()
    # return torch.norm(M).item()


def objective(M: Tensor, W: Tensor, H: Tensor, E: Tensor) -> float:
    return frob_norm(E * (M - W.mm(H))) / frob_norm(E * M)


def proj(A: Tensor, prev: Tensor, sticky: bool):
    res = A.clamp_min(0)
    if not sticky:
        return res
    res[prev < 1e-9] = 0
    return res


def save_img(img: ndarray, name: str):
    import os
    if not os.path.exists("out"):
        os.mkdir("out")
    imsave("out/" + name + ".png", img)
    # imsave("out/" + name + ".png", img.astype(np.uint8))


def init_wh(M: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    (n, m) = M.shape
    avg = sqrt(torch.mean(M) / k)
    W = torch.abs(torch.normal(0, avg, size=(n, k))).to(device)
    H = torch.abs(torch.normal(0, avg, size=(k, m))).to(device)
    return W, H


def to_tensor(M):
    return torch.from_numpy(M).float().to(device)
