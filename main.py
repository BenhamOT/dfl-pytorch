from extractor.extract_frames import extract_frames_from_video
from extractor.extract_faces import extract_faces_from_frames
# from trainer.seahd import SAEHDModel


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    workspace_dir = "workspace"
    data_src_dir = workspace_dir + "/data_src/"
    data_dst_dir = workspace_dir + "/data_dst/"
    data_src_aligned_dir = data_src_dir + "aligned/"
    data_src_landmarks_dir = data_src_dir + "landmarks/"
    data_dst_aligned_dir = data_dst_dir + "aligned/"
    data_dst_landmarks_dir = data_dst_dir + "landmarks/"

    extract_frames_from_video(data_src_dir + "JMT.mp4", data_src_dir, "jpg")
    extract_frames_from_video(data_dst_dir + "MO2.mp4", data_dst_dir, "jpg")

    extract_faces_from_frames(
        input_path=data_src_dir,
        images_output_path=data_src_aligned_dir,
        landmarks_output_path=data_src_landmarks_dir,
        max_faces_from_image=1,
        image_size=512,
        jpeg_quality=100
    )
    extract_faces_from_frames(
        input_path=data_dst_dir,
        images_output_path=data_dst_aligned_dir,
        landmarks_output_path=data_dst_landmarks_dir,
        max_faces_from_image=1,
        image_size=512,
        jpeg_quality=100
    )
