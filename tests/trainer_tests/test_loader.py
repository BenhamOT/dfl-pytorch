import numpy as np
import matplotlib.pyplot as plt
from trainer.data_loader import C2DataLoader
from PIL import Image


def save_tensor_as_image(image_tensor, file_name, mode="RGB"):
    image_array = image_tensor.numpy()

    image_array = np.clip(image_array * 255, 0, 255)

    if mode == "L":
        image_array = image_array[0]
    elif mode == "RGB":
        image_array = image_array.transpose((1, 2, 0)).astype(np.uint8)

    img = Image.fromarray(image_array, mode=mode)
    img.save(file_name)


def display_mask(mask_image):
    plt.imshow(mask_image, cmap='hot', interpolation='nearest')
    plt.show()


def test_data_loader():
    pass


for sample in C2DataLoader(src_path="workspace_test/data_src/", dst_path="workspace_test/data_dst/", batch_size=1).run():

    print("target src mask shape is {}".format(sample["target_src_mask"][0].shape))
    save_tensor_as_image(sample["warped_src"][0], "warped_src.jpg")
    # save_tensor_as_image(sample["target_src"][0], "target_src.jpg")
    save_tensor_as_image(sample["warped_dst"][0], "warped_dst.jpg")
    # save_tensor_as_image(sample["target_dst"][0], "target_dst.jpg")

    np.save("src_mask.npy", sample["target_src_mask"].numpy())
    np.save("dst_mask.npy", sample["target_dst_mask"].numpy())

    display_mask(sample["target_src_mask"][0][0])
    display_mask(sample["target_dst_mask"][0][0])
    break
