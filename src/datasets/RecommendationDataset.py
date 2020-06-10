import statistics
from random import shuffle

import torch
import numpy as np

from torch import Tensor
from data_preprocessors.recommendation_preprocessor import recommentation_dataset
from datasets.Dataset import Dataset


class RecommendationDataset(Dataset):
    def __init__(self):
        self.ratings, self.user_cnt, self.movie_cnt, avg_ratings = recommentation_dataset()
        self.avg_ratings = np.tile(np.array(avg_ratings), (self.user_cnt, 1))
        # self.avg_ratings = np.ones((self.user_cnt, self.movie_cnt)) * statistics.mean(avg_ratings)
        shuffle(self.ratings)
        self.test = self.ratings[len(self.ratings) * 4 // 5:]
        self.compute_error(self.avg_ratings)

    def compute_error(self, arr: np.ndarray):
        sum_error = 0
        for user, movie, rating in self.test:
            sum_error += abs(arr[user, movie] - rating)
        avg_error = sum_error / len(self.test)
        print("Average error:", avg_error.item())

    def generate(self, iteration):
        shuffle(self.ratings)
        self.training = self.ratings[:len(self.ratings) * 4 // 5]
        self.test = self.ratings[len(self.ratings) * 4 // 5:]
        M = self.avg_ratings + 0
        E = np.zeros_like(M)
        for user, movie, rating in self.training:
            M[user, movie] = rating
            E[user, movie] = 1
        M = M * E
        return M, E

    def postprocess(self, W: Tensor, H: Tensor, iteration, name):
        self.compute_error(W.mm(H).cpu().numpy())

    def step(self, iteration):
        return 1e-4

    def iterations(self):
        return 200

    def k(self):
        return 10
