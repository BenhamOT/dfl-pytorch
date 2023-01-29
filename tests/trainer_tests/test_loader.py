import torchshow as ts
import unittest
import random
from trainer.data_loader import CustomDataLoader, CustomImageDataset
from params import Params


class TestImageDS(unittest.TestCase):
    def setUp(self) -> None:
        self.custom_ds = CustomImageDataset()
        self.random_int = random.choice(range(len(self.custom_ds)))

    def test_get_item(self):
        sample = self.custom_ds[self.random_int]
        assert len(sample) == 6
        expected_image_shape = (
            Params.image_input_channels,
            Params.resolution,
            Params.resolution,
        )
        expected_mask_shape = (1, Params.resolution, Params.resolution)
        assert sample[Params.warped_src].shape == expected_image_shape
        assert sample[Params.target_src].shape == expected_image_shape
        assert sample[Params.target_src_mask].shape == expected_mask_shape
        assert sample[Params.warped_dst].shape == expected_image_shape
        assert sample[Params.target_dst].shape == expected_image_shape
        assert sample[Params.target_dst_mask].shape == expected_mask_shape

    def test_create_dst_sublists(self):
        test_dst_dir = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        test_src_dir = ["x", "y", "z"]
        sublists = self.custom_ds.create_dst_sublist(
            dst_dir=test_dst_dir, src_dir=test_src_dir
        )
        assert sublists == [["a", "b", "c", "d"], ["e", "f", "g", "h"], ["i", "j"]]


class TestCustomDL(unittest.TestCase):
    def setUp(self) -> None:
        self.custom_dl = CustomDataLoader()

    def test_run(self):
        for sample in self.custom_dl.run():
            break
        expected_image_shape = (
            Params.batch_size,
            Params.image_input_channels,
            Params.resolution,
            Params.resolution,
        )
        expected_mask_shape = (
            Params.batch_size,
            1,
            Params.resolution,
            Params.resolution,
        )
        assert sample[Params.warped_src].shape == expected_image_shape
        assert sample[Params.target_src].shape == expected_image_shape
        assert sample[Params.target_src_mask].shape == expected_mask_shape
        assert sample[Params.warped_dst].shape == expected_image_shape
        assert sample[Params.target_dst].shape == expected_image_shape
        assert sample[Params.target_dst_mask].shape == expected_mask_shape
