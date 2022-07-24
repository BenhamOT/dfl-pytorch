import torch
import numpy as np
import torch.nn.functional as F


def gaussian(x, mu, sigma):
    return np.exp(-(float(x) - float(mu)) ** 2 / (2 * sigma ** 2))


def make_kernel(sigma):
    kernel_size = max(3, int(2 * 2 * sigma + 1))
    mean = np.floor(0.5 * kernel_size)
    kernel_1d = np.array([gaussian(x, mean, sigma) for x in range(kernel_size)])
    np_kernel = np.outer(kernel_1d, kernel_1d).astype(np.float32)
    kernel = np_kernel / np.sum(np_kernel)
    return kernel, kernel_size


def gaussian_blur(input, radius=2.0):
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


def reducer(x, kernel):
    input_channels = x.shape[1]
    return F.conv2d(x, weight=kernel, stride=1, padding='valid', groups=input_channels)


def dssim(img1, img2, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    filter_size = max(1, filter_size)
    kernel = np.arange(0, filter_size, dtype=np.float32)
    kernel -= (filter_size - 1) / 2.0
    kernel = kernel ** 2
    kernel *= (-0.5 / (filter_sigma ** 2))
    kernel = np.reshape(kernel, (1, -1)) + np.reshape(kernel, (-1, 1))

    kernel = torch.from_numpy(np.reshape(kernel, (1, -1)))
    kernel = F.softmax(kernel, dim=0)
    kernel = torch.reshape(kernel, (1, 1, filter_size, filter_size))
    kernel = torch.tile(kernel, (img1.shape[1], 1, 1, 1))
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    mean0 = reducer(img1, kernel=kernel)
    mean1 = reducer(img2, kernel=kernel)
    num0 = mean0 * mean1 * 2.0

    den0 = torch.square(mean0) + torch.square(mean1)
    luminance = (num0 + c1) / (den0 + c1)

    num1 = reducer(img1 * img2, kernel=kernel) * 2.0
    den1 = reducer(torch.square(img1) + torch.square(img2), kernel=kernel)
    c2 *= 1.0  # compensation factor
    cs = (num1 - num0 + c2) / (den1 - den0 + c2)
    ssim_val = torch.mean(luminance * cs, dim=(2, 3))
    dssim = (1.0 - ssim_val) / 2.0
    return dssim
