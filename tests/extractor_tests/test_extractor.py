import os
import numpy as np
import unittest

from extractor.extract_frames import extract_frames_from_video
from extractor.extract_faces import extract_faces_from_frames
from extractor.utils import (
    pil_loader,
    save_landmarks_on_image,
    mkdir_or_delete_existing_files,
)


class TestExtractFrames(unittest.TestCase):
    def setUp(self) -> None:
        self.input_folder = "workspace_test/extract_frames_test/"
        for image in os.listdir(self.input_folder):
            if image.endswith(".jpg"):
                os.remove(self.input_folder + image)

    def test_extract_frames_from_video(self) -> None:
        expected_number_of_frames = 585
        extract_frames_from_video(
            input_file=self.input_folder + "JMT.mp4",
            output_dir=self.input_folder,
            output_ext="jpg",
        )
        input_image_paths = [
            os.path.join(self.input_folder, x)
            for x in os.listdir(self.input_folder)
            if x.endswith(".jpg")
        ]
        assert len(input_image_paths) == expected_number_of_frames


class TestExtractFace(unittest.TestCase):
    def setUp(self) -> None:
        self.input_folder = "workspace_test/extract_faces_test/"
        self.output_folder = "workspace_test/extract_faces_validation/"
        self.images_folder = self.input_folder + "aligned/"
        self.landmarks_folder = self.input_folder + "landmarks/"

    def test_extract_faces_from_frames(self) -> None:
        number_of_landmarks_files = number_of_faces = 14

        extract_faces_from_frames(
            input_path=self.input_folder,
            images_output_path=self.images_folder,
            landmarks_output_path=self.landmarks_folder,
            max_faces_from_image=1,
            image_size=512,
            jpeg_quality=100,
        )

        # visual validation
        mkdir_or_delete_existing_files(self.output_folder)
        assert len(os.listdir(self.output_folder)) == 0
        for image in os.listdir(self.images_folder):
            img = pil_loader(self.images_folder + image)
            landmarks = np.load(self.landmarks_folder + image.replace(".jpg", ".npy"))
            save_landmarks_on_image(
                image=img, landmarks=landmarks, file_path=self.output_folder + image
            )

        # assert
        assert len(os.listdir(self.output_folder)) == number_of_faces
        assert len(os.listdir(self.images_folder)) == number_of_faces
        assert len(os.listdir(self.landmarks_folder)) == number_of_landmarks_files
