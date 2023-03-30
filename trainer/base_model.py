import torch
from abc import ABC, abstractmethod
from typing import Type, Self, Dict, List


class BaseModelABC(ABC):
    """
    Base class that defines the structure of any model classes that needs to be created for training
    """

    @abstractmethod
    def train(self, sample: Dict[str, torch.tensor]) -> List[torch.tensor]:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> Type[Self]:
        raise NotImplementedError
