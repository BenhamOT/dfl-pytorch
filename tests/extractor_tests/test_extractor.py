import os
import numpy as np

from extractor.extract_frames import extract_frames_from_video
from extractor.extract_faces import extract_faces_from_frames
from extractor.utils import pil_loader, save_landmarks_on_image, mkdir_or_delete_existing_files


def test_extract_frames():
    # clear directory
    input_folder = "workspace_test/extract_frames_test/"
    for image in os.listdir(input_folder):
        if image.endswith(".jpg"):
            os.remove(input_folder + image)

    # assembled
    expected_number_of_frames = 585

    # act
    extract_frames_from_video(
        input_file=input_folder + "JMT.mp4",
        output_dir=input_folder,
        output_ext="jpg"
    )
    input_image_paths = [os.path.join(input_folder, x) for x in os.listdir(input_folder) if x.endswith(".jpg")]

    # assert
    assert len(input_image_paths) == expected_number_of_frames


def test_extract_faces():
    input_folder = "workspace_test/extract_faces_test/"
    output_folder = "workspace_test/extract_faces_validation/"
    images_folder = input_folder + "aligned/"
    landmarks_folder = input_folder + "landmarks/"

    # assemble
    number_of_landmarks_files = number_of_faces = 14

    # act
    extract_faces_from_frames(
        input_path=input_folder,
        images_output_path=images_folder,
        landmarks_output_path=landmarks_folder,
        max_faces_from_image=1,
        image_size=512,
        jpeg_quality=100
    )

    # visual validation
    mkdir_or_delete_existing_files(output_folder)
    assert len(os.listdir(output_folder)) == 0
    for image in os.listdir(images_folder):
        img = pil_loader(images_folder + image)
        landmarks = np.load(landmarks_folder + image.rstrip(".jpg") + ".npy")
        save_landmarks_on_image(
            image=img,
            landmarks=landmarks,
            file_path=output_folder+ image
        )


    # assert
    assert len(os.listdir(output_folder)) == number_of_faces
    assert len(os.listdir(images_folder)) == number_of_faces
    assert len(os.listdir(landmarks_folder)) == number_of_landmarks_files

