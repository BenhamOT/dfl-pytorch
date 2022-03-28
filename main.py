from time import time
from extractor.extract_frames import extract_frames_from_video
from extractor.extract_faces import extract_faces_from_frames
# from trainer.seahd import SAEHDModel


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # extract_frames_from_video("workspace/data_dst/HS.mp4", "workspace/data_dst", "jpg")
    workspace_dir = "workspace_test"
    # data_src_dir
    # data_dst_dir
    # data_src_aligned_dir
    # data_src_landmarks_dir
    #
    extract_faces_from_frames(
        input_path="workspace_test/data_dst/",
        images_output_path="workspace_test/data_dst/aligned/",
        landmarks_output_path="workspace_test/data_dst/landmarks/",
        max_faces_from_image=1,
        image_size=512,
        jpeg_quality=100
    )
    extract_faces_from_frames(
        input_path="workspace_test/data_src/",
        images_output_path="workspace_test/data_src/aligned/",
        landmarks_output_path="workspace_test/data_src/landmarks/",
        max_faces_from_image=1,
        image_size=512,
        jpeg_quality=100
    )
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
