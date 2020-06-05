import numpy as np
import utils
from numpy import ndarray
import sklearn


class LeastSquares:
    def __init__(self, M: ndarray, k: int):
        self.M = M
        self.k = k
        self.name = "HALS"

    def optimize(self, step, iterations, save_image):
        (n, m) = self.M.shape
        M = self.M
        W = np.random.uniform(0, 1, (n, self.k))
        H = np.random.uniform(0, 1, (self.k, m))
        for j in range(self.k):
            W[:, j] = W[:, j] / np.linalg.norm(W[:, j])
            H[j, :] = H[j, :] / np.linalg.norm(H[j, :])
        for i in range(iterations):
            print(i, utils.objective(M, W, H))
            E: ndarray = M - W.dot(H)
            for j in range(self.k):
                # print(W[:, j])
                # print(H[j, :])
                # E = E + W[:, j].dot(H[j, :].transpose())
                uj = W[:, j]
                vj = H[j, :]
                E = E + uj.reshape(len(uj), 1).dot(vj.reshape(1, len(vj)))
                # E = M - W.dot(H) + uj.reshape(len(uj), 1).dot(vj.reshape(1, len(vj)))

                u_nom = E.transpose().dot(uj).clip(0)
                if np.linalg.norm(u_nom) > 1e-18:
                    vj = u_nom / (np.linalg.norm(uj) ** 2)
                else:
                    vj = np.zeros_like(vj)

                v_nom = E.dot(vj).clip(0)
                if np.linalg.norm(v_nom) > 1e-18:
                    uj = v_nom / (np.linalg.norm(vj) ** 2)
                else:
                    uj = np.zeros_like(uj)
                W[:, j] = uj
                H[j, :] = vj

                # W[:, j] = W[:, j] / np.linalg.norm(W[:, j])
                E = E - uj.reshape(len(uj), 1).dot(vj.reshape(1, len(vj)))
        # for i in range(iterations):
        #     A = M.dot(H)
        #     B = H.dot(H.transpose())
        #     for j in range(self.k):

        if save_image:
            utils.save_img(W.dot(H), 'hals')
        print()
        return utils.objective(M, W, H)