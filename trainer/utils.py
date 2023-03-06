import torch
import numpy as np
import torch.nn.functional as F
from typing import Tuple


def gaussian(x: int, mu: int, sigma: int) -> float:
    return np.exp(-((float(x) - float(mu)) ** 2) / (2 * sigma**2))


def make_kernel(sigma: int) -> Tuple[np.array, int]:
    kernel_size = max(3, int(2 * 2 * sigma + 1))
    mean = np.floor(0.5 * kernel_size)
    kernel_1d = np.array([gaussian(x, mean, sigma) for x in range(kernel_size)])
    np_kernel = np.outer(kernel_1d, kernel_1d).astype(np.float32)
    kernel = np_kernel / np.sum(np_kernel)
    return kernel, kernel_size


def gaussian_blur(input: torch.tensor, radius: int = 2.0) -> torch.tensor:
    """
    Used to produce a blur effect on the input image tensor.
    """
    gauss_kernel, kernel_size = make_kernel(radius)
    padding = kernel_size // 2
    if padding != 0:
        padding = (padding, padding, padding, padding, 0, 0, 0, 0)
    else:
        padding = None
    gauss_kernel = gauss_kernel[None, None, :, :]

    channels = input.shape[1]
    x = input
    k = torch.tile(torch.from_numpy(gauss_kernel), (channels, 1, 1, 1))
    x = F.pad(x, padding)
    x = F.conv2d(x, weight=k, stride=1, padding="valid", groups=channels)
    return x
