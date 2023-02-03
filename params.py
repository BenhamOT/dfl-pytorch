import cv2


class Params:
    # file path settings
    src_video_name = "JMT.mp4"
    dst_video_name = "MO2.mp4"
    workspace_dir = "workspace"
    data_src_dir = workspace_dir + "/data_src/"
    data_dst_dir = workspace_dir + "/data_dst/"
    src_video_file_path = data_src_dir + src_video_name
    dst_video_file_path = data_dst_dir + dst_video_name
    data_src_aligned_dir = data_src_dir + "aligned/"
    data_src_landmarks_dir = data_src_dir + "landmarks/"
    data_dst_aligned_dir = data_dst_dir + "aligned/"
    data_dst_landmarks_dir = data_dst_dir + "landmarks/"

    # extraction settings
    max_faces_from_image = (1,)
    image_size = (512,)
    jpeg_quality = 100
    image_extension = ".jpg"

    # data loader settings
    border_mode = cv2.BORDER_REPLICATE
    warp = False
    batch_size = 4

    # training settings
    resolution = 128
    e_dims = 80
    ae_dims = 128
    d_dims = 48
    d_mask_dims = 16
    masked_training = True
    # learn_mask = should this ever be false - need to check with dfl
    eyes_priority = False
    # lr_dropout =
    # random_warp =
    # target_iterations =
    # random_flip =
    # batch_size =
    # pretrain =
    # uniform_yaw =
    # ct_mode =
    # clip_gradients =
    is_training = True
    epochs = 100
    image_input_channels = 3

    # data loader setting
    warped_src = "warped_src"
    target_src = "target_src"
    target_src_mask = "target_src_mask"
    warped_dst = "warped_dst"
    target_dst = "target_dst"
    target_dst_mask = "target_dst_mask"
