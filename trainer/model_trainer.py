from trainer.data_loader import CustomDataLoader
from params import Params
from typing import Type
from base_model import BaseModelABC


class ModelTrainer:
    """
    Could include some MLOps here
    """

    def __init__(self, model: Type[BaseModelABC], epochs: int = Params.epochs):
        self.epochs = epochs
        self.model = model

    def run(self):
        for i in range(self.epochs):
            print("epoch {}".format(i))

            # TODO add in tqdm here when finsihed debugging
            for sample in CustomDataLoader().run():
                loss = self.train(sample=sample)
                print(loss)
                # if len(loss) > 1: ?
                # print("src loss is {}".format(src_loss))
                # print("dst loss is {}".format(dst_loss))
                # print("combined loss is {}".format(combined_loss))

    def train(self, sample):
        loss = self.model.train(sample)
        return loss

    def save(self, model, path):
        pass

    def load(self, model, path):
        pass
