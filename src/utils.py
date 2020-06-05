import numpy as np
from numpy import ndarray
from skimage.io import imsave


def frob_norm(M: ndarray) -> float:
    return np.linalg.norm(M, 'fro')


def objective(M: ndarray, W: ndarray, H: ndarray) -> float:
    return frob_norm(M - W.dot(H)) / frob_norm(M)


def proj(A: ndarray, prev: ndarray, sticky: bool):
    if not sticky:
        return A.clip(0)
    res = A.clip(0)
    res[prev < 1e-9] = 0
    return res


def save_img(img: ndarray, name: str):
    import os
    if not os.path.exists("out"):
        os.mkdir("out")
    imsave("out/" + name + ".png", img)
