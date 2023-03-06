import os
import cv2
import numpy as np
import torch
import math
import random
from typing import Dict, List
from torch.utils.data import DataLoader, Dataset
from extractor.utils import pil_loader
from extractor.landmarks_processor import (
    get_image_hull_mask,
    get_image_eye_mask,
)
from trainer.warp_preprocessing import warp_by_params, gen_warp_params
from params import Params


class CustomImageDataset(Dataset):
    def __init__(self) -> None:
        # define src and dst image and landmark directories
        self.data_src_aligned_dir = Params.data_src_aligned_dir
        self.data_dst_aligned_dir = Params.data_dst_aligned_dir
        self.data_src_landmarks_dir = Params.data_src_landmarks_dir
        self.data_dst_landmarks_dir = Params.data_dst_landmarks_dir

        # get a list of the src and dst image files
        self.src_dir = os.listdir(self.data_src_aligned_dir)
        self.dst_dir = os.listdir(self.data_dst_aligned_dir)
        self.dst_dir_sublists = self.create_dst_sublist(
            dst_dir=self.dst_dir, src_dir=self.src_dir
        )

        # define the settings
        self.resolution = Params.resolution
        self.border_mode = Params.border_mode
        self.warp = Params.warp
        self.params = None

    def __len__(self) -> int:
        return len(self.src_dir)

    def get_face_image(
        self,
        img: np.ndarray,
        warp: bool,
        warp_affine_flags=cv2.INTER_CUBIC,
        masked: bool = False,
    ) -> torch.tensor:
        img = cv2.resize(
            img, (self.resolution, self.resolution), interpolation=warp_affine_flags
        )
        img = warp_by_params(
            params=self.params,
            img=img,
            random_warp=warp,
            transform=True,
            can_flip=True,
            border_mode=self.border_mode,
            cv2_inter=warp_affine_flags,
        )
        if not masked:
            img = np.clip(img.astype(np.float32), 0, 1)

        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img)

    def get_face_mask(self, img: np.ndarray, landmarks: np.ndarray) -> torch.tensor:
        full_face_mask = get_image_hull_mask(img.shape, landmarks)
        img = np.clip(full_face_mask, 0, 1)

        eyes_mask = get_image_eye_mask(img.shape, landmarks)
        clipped_eye_mask = np.clip(eyes_mask, 0, 1)
        img += clipped_eye_mask * img

        img = self.get_face_image(
            img=img, warp=False, warp_affine_flags=cv2.INTER_LINEAR, masked=True
        )
        return img

    @staticmethod
    def create_dst_sublist(dst_dir: List[str], src_dir: List[str]) -> List[List]:
        len_of_sublist = math.ceil(len(dst_dir) / len(src_dir))

        dst_dir_copy = dst_dir.copy()
        list_of_sublists, sublist = [], []

        for file_name in dst_dir:
            if len(sublist) < len_of_sublist:
                sublist.append(file_name)

            if len(sublist) == len_of_sublist:
                list_of_sublists.append(sublist)
                sublist = []

            dst_dir_copy.pop(0)
            if len(list_of_sublists) + len(dst_dir_copy) + bool(sublist) == len(
                src_dir
            ):
                if sublist:
                    list_of_sublists.append(sublist)
                break

        list_of_sublists.extend([i] for i in dst_dir_copy)
        return list_of_sublists

    def get_destination_image_path(self, item: int) -> str:
        sublist = self.dst_dir_sublists[item]
        return random.choice(sublist)

    def __getitem__(self, item: int) -> Dict:
        self.params = gen_warp_params(w=self.resolution)
        src_image_file = self.src_dir[item]
        src_landmarks_file = src_image_file.replace(Params.image_extension, ".npy")
        dst_image_file = self.get_destination_image_path(item)
        dst_landmarks_file = dst_image_file.replace(Params.image_extension, ".npy")
        src_image = pil_loader(
            self.data_src_aligned_dir + src_image_file, normalise=True
        )
        src_landmarks = np.load(self.data_src_landmarks_dir + src_landmarks_file)
        dst_image = pil_loader(
            self.data_dst_aligned_dir + dst_image_file, normalise=True
        )
        dst_landmarks = np.load(self.data_dst_landmarks_dir + dst_landmarks_file)

        # create the warped, target and target mask for the src image
        warped_src = self.get_face_image(img=src_image.copy(), warp=self.warp)
        target_src = self.get_face_image(img=src_image.copy(), warp=False)
        target_src_mask = self.get_face_mask(
            img=src_image.copy(), landmarks=src_landmarks
        )

        # create the  warped, target and target mask for the dst image
        warped_dst = self.get_face_image(img=dst_image.copy(), warp=self.warp)
        target_dst = self.get_face_image(img=dst_image.copy(), warp=False)
        target_dst_mask = self.get_face_mask(
            img=dst_image.copy(), landmarks=dst_landmarks
        )

        result = {
            Params.warped_src: warped_src,
            Params.target_src: target_src,
            Params.target_src_mask: target_src_mask,
            Params.warped_dst: warped_dst,
            Params.target_dst: target_dst,
            Params.target_dst_mask: target_dst_mask,
        }
        return result


class CustomDataLoader:
    def __init__(self, batch_size: int = Params.batch_size) -> None:
        self.batch_size = batch_size

    def run(self) -> DataLoader:
        print("loading image folder")
        data = CustomImageDataset()
        print("creating image loader")
        data = DataLoader(data, self.batch_size, shuffle=False)
        return data
