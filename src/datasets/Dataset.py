from abc import ABC, abstractmethod
from typing import Tuple

from numpy import ndarray


class Dataset(ABC):
    @abstractmethod
    def generate(self, iteration) -> Tuple[ndarray, ndarray]:
        pass

    @abstractmethod
    def postprocess(self, W, H, iteration, name):
        pass

    @abstractmethod
    def step(self, iteration: int) -> float:
        pass

    @abstractmethod
    def iterations(self) -> int:
        pass

    @abstractmethod
    def k(self) -> int:
        pass
