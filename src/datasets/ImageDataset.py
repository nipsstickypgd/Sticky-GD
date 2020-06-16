from torch import Tensor

import utils
import numpy as np
from data_preprocessors.image_preprocessor import read_image
from datasets.Dataset import Dataset


class ImageDataset(Dataset):
    def generate(self, iteration):
        M = read_image("kodak_" + str(iteration))
        # M = read_image("0000_0000000" + str(iteration + 1))
        return M, np.ones_like(M)

    def postprocess(self, W: Tensor, H: Tensor, iteration, name):
        # if iteration == 0:
        utils.save_img(W.mm(H).cpu().numpy(), name + "_" + str(iteration))

    def step(self, iteration):
        # return 5e-6
        return 1e-5

    def iterations(self):
        return 200

    def k(self):
        return 100
