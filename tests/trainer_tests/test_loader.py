import torchshow as ts
import unittest
from trainer.data_loader import CustomDataLoader, CustomImageDataset


class TestImageDS:

    def __init__(self):
        self.custom_ds = CustomImageDataset(
            src_path="workspace_test/extract_faces_test/",
            dst_path="workspace_test/extract_faces_test/",
        )

    def test_get_face_image(self):
        pass

    def test_get_face_mask(self):
        pass


class TestCustomDL:
    pass





# for sample in C2DataLoader(src_path="workspace_test/data_src/", dst_path="workspace_test/data_dst/",
#                            batch_size=1).run():
#
#     ts.show(sample["warped_src"])
#     ts.show(sample["warped_dst"])
#     ts.show(sample["target_src_mask"])
#     ts.show(sample["target_dst_mask"])
#     break
