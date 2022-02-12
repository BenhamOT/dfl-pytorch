import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from typing import Optional, Callable
from extractor.utils import pil_loader


class C2CustomImageDataset(Dataset):
    def __init__(self, workspace_directory: str, transform: Optional[Callable]):
        self.data_src_aligned_dir = workspace_directory + "data_src/aligned/"
        self.data_dst_aligned_dir = workspace_directory + "data_dst/aligned/"
        self.src_dir = os.listdir(self.data_src_aligned_dir)
        self.dst_dir = os.listdir(self.data_dst_aligned_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.src_dir)

    def __getitem__(self, item: int):

        src_file = self.src_dir[item]
        dst_file = self.dst_dir[item]  # need a way to randomly choose a target image from a subnet of target images
        src_image = pil_loader(self.data_src_aligned_dir + src_file)
        dst_image = pil_loader(self.data_dst_aligned_dir + dst_file)

        result = {"src": self.transform(src_image), "dst": self.transform(dst_image)}
        return result


class C2DataLoader:

    def __init__(self, src_path=None, dst_path=None, batch_size=4):
        self.src_path = src_path
        self.dst_path = dst_path
        self.batch_size = batch_size

    def run(self):
        print("loading image folder")
        transform = transforms.Compose([
            # you can add other transformations in this list
            transforms.ToTensor(),
            # transforms.Resize((128, 128))
        ])
        # data = ImageFolder(root="../workspace/training_data/", transform=transform)
        data = C2CustomImageDataset(workspace_directory="../workspace/", transform=transform)
        print("creating image loader")
        data = DataLoader(data, self.batch_size, shuffle=False)
        return data


# loader = C2DataLoader().run()
#
# for sample in loader:
#     print(sample["src"].size(), sample["dst"].size())


