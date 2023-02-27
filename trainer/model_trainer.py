import torch

from trainer.data_loader import CustomDataLoader
from params import Params
from typing import Type, Dict, List
from base_model import BaseModelABC


class ModelTrainer:
    """
    Could include some MLOps here
    """

    def __init__(self, model: Type[BaseModelABC], epochs: int = Params.epochs) -> None:
        self.epochs = epochs
        self.model = model

    def run(self) -> None:
        for i in range(self.epochs):
            print("epoch {}".format(i))

            # TODO add in tqdm here when finsihed debugging
            for sample in CustomDataLoader().run():
                loss = self.train(sample=sample)
                print(loss)

    def train(self, sample: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        loss = self.model.train(sample)
        return loss

    def save(self, model: Type[BaseModelABC], path: str) -> None:
        pass

    def load(self, model: Type[BaseModelABC], path: str) -> Type[BaseModelABC]:
        pass
