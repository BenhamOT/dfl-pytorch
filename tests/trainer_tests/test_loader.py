import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from trainer.data_loader import C2DataLoader


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


class TestC2DataLoader:
    pass
