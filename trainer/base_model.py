from abc import ABC


class BaseModelABC(ABC):
    """
    Base class that defines the structure of any model classes that needs to be created for training
    """

    @staticmethod
    def train(sample):
        pass

    @staticmethod
    def save(path):
        pass

    @staticmethod
    def load(path):
        pass
