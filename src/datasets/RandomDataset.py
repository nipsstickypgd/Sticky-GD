from datasets.Dataset import Dataset
import numpy as np


class RandomDataset(Dataset):
    def __init__(self):
        self.E = np.ones((1000, 1000))

    def generate(self, iteration):
        return np.random.uniform(0, 100, (1000, 1000)), self.E

    def postprocess(self, W, H, iteration, name):
        pass

    def step(self, iteration):
        return 1e-5

    def iterations(self):
        return 100

    def k(self):
        return 10