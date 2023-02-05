from abc import ABC


class BaseModelABC(ABC):
    """
    Base class that defines the structure of any model classes that needs to be created for training
    """

    @staticmethod
    def train(sample) -> None:
        pass

    @staticmethod
    def save(path) -> None:
        pass

    @staticmethod
    def load(path) -> None:
        pass
