from params import Params
from extractor.extract_frames import extract_frames_from_video
from extractor.extract_faces import extract_faces_from_frames
from trainer.seahd import SAEHDModel


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # extract_frames_from_video(Params.src_video_file_path, Params.data_src_dir, "jpg")
    # extract_frames_from_video(Params.dst_video_file_path, Params.data_dst_dir, "jpg")

    # extract_faces_from_frames(
    #     input_path=Params.data_src_dir,
    #     images_output_path=Params.data_src_aligned_dir,
    #     landmarks_output_path=Params.data_src_landmarks_dir,
    #     max_faces_from_image=Params.max_faces_from_image,
    #     image_size=Params.image_size,
    #     jpeg_quality=Params.jpeg_quality
    # )
    # extract_faces_from_frames(
    #     input_path=Params.data_dst_dir,
    #     images_output_path=Params.data_dst_aligned_dir,
    #     landmarks_output_path=Params.data_dst_landmarks_dir,
    #     max_faces_from_image=Params.max_faces_from_image,
    #     image_size=Params.image_size,
    #     jpeg_quality=Params.jpeg_quality
    # )

    saehd = SAEHDModel()
    saehd.run(src_path=Params.data_src_dir, dst_path=Params.data_dst_dir)

