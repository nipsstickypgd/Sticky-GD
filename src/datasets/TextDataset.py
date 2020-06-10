import numpy as np
from sklearn.metrics import precision_score

from torch import Tensor
from data_preprocessors.text_preprocessor import reuters_dataset
from datasets.Dataset import Dataset
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier


class TextDataset(Dataset):
    def __init__(self):
        self.train_data, self.test_data, self.train_labels, self.test_labels = reuters_dataset()
        self.data = np.concatenate([self.train_data, self.test_data])
        self.E = np.ones_like(self.data)

    def generate(self, iteration):
        return self.data, self.E

    def postprocess(self, W: Tensor, H: Tensor, iteration, name):
        W = W.cpu().numpy()
        classifier = OneVsRestClassifier(LinearSVC(random_state=42))
        classifier.fit(W[:len(self.train_data), :], self.train_labels)

        predictions = classifier.predict(W[len(self.train_data):, :])
        precision = precision_score(self.test_labels, predictions, average='micro')
        print("Precision:", precision)

    def step(self, iteration):
        return 0.05

    def iterations(self):
        return 100

    def k(self):
        return 10
