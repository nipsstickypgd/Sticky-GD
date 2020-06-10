from abc import ABC, abstractmethod
from typing import Tuple

from numpy import ndarray
from torch import Tensor


class Optimizer(ABC):
    @abstractmethod
    def optimize(self, E: Tensor, iterations: int) -> Tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def name(self):
        pass

    def short_name(self):
        return self.name()
