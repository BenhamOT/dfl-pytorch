import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import List

from params import Params
from extractor.utils import pil_loader, mkdir_or_delete_existing_files
from extractor.s3fd import SFDDetector
from extractor.fan2d import FaceAlignment
from extractor.landmarks_processor import get_transform_mat, transform_points


class ExtractData:
    def __init__(
        self,
        filepath: str = None,
        rects: List[np.array] = None,
        landmarks: List[np.array] = None,
        final_output_files: List[str] = None,
    ) -> None:
        self.filepath = filepath
        self.file_name = filepath.split("/")[-1].replace(Params.image_extension, "")
        self.rects = rects or []
        self.rects_rotation = 0
        self.landmarks = landmarks or []
        self.final_output_files = final_output_files or []
        self.faces_detected = 0


class ExtractFaces:
    def __init__(
        self,
        input_data: List[str],
        image_size: int,
        images_output_path: str,
        landmarks_output_path: str,
        max_faces_from_image: int = 0,
    ):

        self.input_data = input_data
        self.image_size = image_size
        self.max_faces_from_image = max_faces_from_image
        self.images_output_path = images_output_path
        self.landmarks_output_path = landmarks_output_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Device = {}".format(self.device))
        self.rects_extractor = SFDDetector(
            device=self.device, path_to_detector="extractor/models/s3fd.pth"
        )
        self.landmarks_extractor = FaceAlignment(
            device=self.device,
            path_to_detector="extractor/models/face-alignment-net.pt",
        )
        self.detected_faces = None

    def run(self) -> None:
        self.detected_faces = 0
        for image_file_path in tqdm(self.input_data):
            data = ExtractData(filepath=image_file_path)
            data = self.process_data(data)
            self.detected_faces += data.faces_detected

    def process_data(self, data: ExtractData) -> ExtractData:
        filepath = data.filepath
        image = pil_loader(filepath)

        data = self.rects_stage(
            data=data,
            image=image.copy(),
            max_faces_from_image=self.max_faces_from_image,
            rects_extractor=self.rects_extractor,
        )
        data = self.landmarks_stage(
            data=data, image=image.copy(), landmarks_extractor=self.landmarks_extractor
        )
        data = self.final_stage(data=data, image=image, image_size=self.image_size)
        return data

    @staticmethod
    def rects_stage(
        data: ExtractData,
        image: np.ndarray,
        max_faces_from_image: int,
        rects_extractor: SFDDetector,
    ) -> ExtractData:
        h, w, c = image.shape
        if min(h, w) < 128:
            # Image is too small
            data.rects = []
        else:
            data.rects = rects_extractor.detect_from_image(image)
            if max_faces_from_image > 0 and len(data.rects) > 0:
                data.rects = data.rects[0:max_faces_from_image]

        return data

    @staticmethod
    def landmarks_stage(
        data: ExtractData, image: np.ndarray, landmarks_extractor: FaceAlignment
    ) -> ExtractData:
        if not data.rects:
            return data

        data.landmarks = landmarks_extractor.get_landmarks_from_image(image, data.rects)
        return data

    def final_stage(
        self, data: ExtractData, image: np.ndarray, image_size: int
    ) -> ExtractData:
        data.final_output_files = []
        file_name = data.file_name
        rects = data.rects
        landmarks = data.landmarks
        if landmarks is None:
            return data

        face_idx = 0
        for rect, image_landmarks in zip(rects, landmarks):
            image_to_face_mat = get_transform_mat(image_landmarks, image_size)
            face_image = cv2.warpAffine(
                image, image_to_face_mat, (image_size, image_size), cv2.INTER_LANCZOS4
            )
            face_image = Image.fromarray(face_image)
            # save the image
            images_output_filepath = (
                self.images_output_path
                + f"{file_name}_{face_idx}{Params.image_extension}"
            )
            face_image.save(images_output_filepath)
            # save the landmakrs
            face_image_landmarks = transform_points(
                points=image_landmarks, mat=image_to_face_mat
            )
            landmarks_output_filepath = (
                self.landmarks_output_path + f"{file_name}_{face_idx}.npy"
            )
            np.save(landmarks_output_filepath, face_image_landmarks)

            data.final_output_files.append(images_output_filepath)
            face_idx += 1

        data.faces_detected = face_idx
        return data


def extract_faces_from_frames(
    input_path: str,
    images_output_path: str,
    landmarks_output_path: str,
    image_size: int,
    max_faces_from_image: int = 0,
) -> None:
    input_image_paths = [
        os.path.join(input_path, x)
        for x in os.listdir(input_path)
        if x.endswith(Params.image_extension)
    ]

    # delete files from aligned or landmarks dir if it's not empty
    mkdir_or_delete_existing_files(path=images_output_path)
    mkdir_or_delete_existing_files(path=landmarks_output_path)

    print("Extracting faces...")
    extract_faces = ExtractFaces(
        input_image_paths,
        image_size,
        max_faces_from_image=max_faces_from_image,
        images_output_path=images_output_path,
        landmarks_output_path=landmarks_output_path,
    )
    extract_faces.run()

    print("-------------------------")
    print("Images found: {}".format(len(input_image_paths)))
    print("Faces detected: {}".format(extract_faces.detected_faces))
    print("-------------------------")
