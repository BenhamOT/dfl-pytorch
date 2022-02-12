from time import time
from extractor.extract_frames import extract_frames_from_video
from extractor.extract_faces import extract_faces_from_frames


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # extract_frames_from_video("workspace/data_dst/HS.mp4", "workspace/data_dst", "jpg")
    start = time()
    extract_faces_from_frames(
        input_path="workspace_test/data_src/",
        images_output_path="workspace_test/data_src/aligned/",
        landmarks_output_path="workspace_test/data_src/landmarks/",
        max_faces_from_image=0,
        image_size=512,
        jpeg_quality=100

    )
    end = time()
    print("Total time taken is {}".format(end-start))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
