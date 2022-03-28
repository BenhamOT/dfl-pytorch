import os
import sys
import errno
import torch
import numpy as np
import cv2
from PIL import Image
from numba import jit
from urllib.parse import urlparse
from torch.hub import download_url_to_file, HASH_REGEX
try:
    from torch.hub import get_dir
except BaseException:
    from torch.hub import _get_torch_home as get_dir


def pil_loader(path: str, normalise=False) -> np.ndarray:
    """
    Load image as RGB numpy array

    Arguments:
         path {str} -- the input image file path
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = np.asarray(img.convert('RGB'))  # /255.0
        if normalise:
            img = img/255.0
        return img


def mkdir_or_delete_existing_files(path):
    """
    Make a new directory for images or landmark data files or delete files from an existing directory

    Args:
        path {str} -- the directory path
    """
    if os.path.isdir(path):
        for file in os.listdir(path):
            os.remove(path + file)
    else:
        os.mkdir(path)


def transform(point, center, scale, resolution, invert=False):
    """
    Generate and affine transformation matrix.
    Given a set of points, a center, a scale and a targer resolution, the
    function generates and affine transformation matrix. If invert is ``True``
    it will produce the inverse transformation.
    Arguments:
        point {torch.tensor} -- the input 2D point
        center {torch.tensor or numpy.array} -- the center around which to perform the transformations
        scale {float} -- the scale of the face/object
        resolution {float} -- the output resolution
    Keyword Arguments:
        invert {bool} -- define wherever the function should produce the direct or the
        inverse transformation matrix (default: {False})
    """
    _pt = torch.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = torch.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = torch.inverse(t)

    new_point = (torch.matmul(t, _pt))[0:2]

    return new_point.int()


def crop(image, center, scale, resolution=256.0):
    """
    Center crops an image or set of heatmaps
    Arguments:
        image {numpy.array} -- an rgb image
        center {numpy.array} -- the center of the object, usually the same as of the bounding box
        scale {float} -- scale of the face
    Keyword Arguments:
        resolution {float} -- the size of the output cropped image (default: {256.0})
    Returns:
        [type] -- [description]
    """
    ul = transform([1, 1], center, scale, resolution, True)
    br = transform([resolution, resolution], center, scale, resolution, True)
    # pad = math.ceil(torch.norm((ul - br).float()) / 2.0 - (br[0] - ul[0]) / 2.0)
    if image.ndim > 2:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0],
                           image.shape[2]], dtype=np.int32)
        newImg = np.zeros(newDim, dtype=np.uint8)
    else:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
        newImg = np.zeros(newDim, dtype=np.uint8)
    ht = image.shape[0]
    wd = image.shape[1]
    newX = np.array(
        [max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
    newY = np.array(
        [max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
    oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
    oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
    newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1]
           ] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]
    newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)),
                        interpolation=cv2.INTER_LINEAR)
    return newImg


@jit(nopython=True)
def transform_np(point, center, scale, resolution, invert=False):
    """
    Generate and affine transformation matrix.
    Given a set of points, a center, a scale and a targer resolution, the
    function generates and affine transformation matrix. If invert is ``True``
    it will produce the inverse transformation.
    Arguments:
        point {numpy.array} -- the input 2D point
        center {numpy.array} -- the center around which to perform the transformations
        scale {float} -- the scale of the face/object
        resolution {float} -- the output resolution
    Keyword Arguments:
        invert {bool} -- define wherever the function should produce the direct or the
        inverse transformation matrix (default: {False})
    """
    _pt = np.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = np.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = np.ascontiguousarray(np.linalg.pinv(t))

    new_point = np.dot(t, _pt)[0:2]

    return new_point.astype(np.int32)


def get_preds_fromhm(hm, center=None, scale=None):
    """
    Obtain (x,y) coordinates given a set of N heatmaps. If the center
    and the scale is provided the function will return the points also in
    the original coordinate frame.
    Arguments:
        hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, W, H]
    Keyword Arguments:
        center {torch.tensor} -- the center of the bounding box (default: {None})
        scale {float} -- face scale (default: {None})
    """
    B, C, H, W = hm.shape
    hm_reshape = hm.reshape(B, C, H * W)
    idx = np.argmax(hm_reshape, axis=-1)
    scores = np.take_along_axis(hm_reshape, np.expand_dims(idx, axis=-1), axis=-1).squeeze(-1)
    preds, preds_orig = _get_preds_fromhm(hm, idx, center, scale)

    return preds, preds_orig, scores


@jit(nopython=True)
def _get_preds_fromhm(hm, idx, center=None, scale=None):
    """
    Obtain (x,y) coordinates given a set of N heatmaps and the
    coresponding locations of the maximums. If the center
    and the scale is provided the function will return the points also in
    the original coordinate frame.
    Arguments:
        hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, W, H]
    Keyword Arguments:
        center {torch.tensor} -- the center of the bounding box (default: {None})
        scale {float} -- face scale (default: {None})
    """
    B, C, H, W = hm.shape
    idx += 1
    preds = idx.repeat(2).reshape(B, C, 2).astype(np.float32)
    preds[:, :, 0] = (preds[:, :, 0] - 1) % W + 1
    preds[:, :, 1] = np.floor((preds[:, :, 1] - 1) / H) + 1

    for i in range(B):
        for j in range(C):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = np.array(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j] += np.sign(diff) * 0.25

    preds -= 0.5

    preds_orig = np.zeros_like(preds)
    if center is not None and scale is not None:
        for i in range(B):
            for j in range(C):
                preds_orig[i, j] = transform_np(
                    preds[i, j], center, scale, H, True)

    return preds, preds_orig


def create_bounding_box(target_landmarks, expansion_factor=0.0):
    """
    gets a batch of landmarks and calculates a bounding box that includes all the landmarks per set of landmarks in
    the batch

    Arguments:
        target_landmarks: batch of landmarks of dim (n x 68 x 2). Where n is the batch size
        expansion_factor: expands the bounding box by this factor. For example, a `expansion_factor` of 0.2 leads
            to 20% increase in width and height of the boxes
    Returns:
        a batch of bounding boxes of dim (n x 4) where the second dim is (x1,y1,x2,y2)
    """
    # Calc bounding box
    x_y_min, _ = target_landmarks.reshape(-1, 68, 2).min(dim=1)
    x_y_max, _ = target_landmarks.reshape(-1, 68, 2).max(dim=1)
    # expanding the bounding box
    expansion_factor /= 2
    bb_expansion_x = (x_y_max[:, 0] - x_y_min[:, 0]) * expansion_factor
    bb_expansion_y = (x_y_max[:, 1] - x_y_min[:, 1]) * expansion_factor
    x_y_min[:, 0] -= bb_expansion_x
    x_y_max[:, 0] += bb_expansion_x
    x_y_min[:, 1] -= bb_expansion_y
    x_y_max[:, 1] += bb_expansion_y
    return torch.cat([x_y_min, x_y_max], dim=1)


def shuffle_lr(parts, pairs=None):
    """
    Shuffle the points left-right according to the axis of symmetry
    of the object.
    Arguments:
        parts {torch.tensor} -- a 3D or 4D object containing the
        heatmaps.
    Keyword Arguments:
        pairs {list of integers} -- [order of the flipped points] (default: {None})
    """
    if pairs is None:
        pairs = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 27, 28, 29, 30, 35,
                 34, 33, 32, 31, 45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41,
                 40, 54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55, 64, 63,
                 62, 61, 60, 67, 66, 65]
    if parts.ndimension() == 3:
        parts = parts[pairs, ...]
    else:
        parts = parts[:, pairs, ...]

    return parts


def flip(tensor, is_label=False):
    """
    Flip an image or a set of heatmaps left-right
    Arguments:
        tensor {numpy.array or torch.tensor} -- [the input image or heatmaps]
    Keyword Arguments:
        is_label {bool} -- [denote wherever the input is an image or a set of heatmaps ] (default: {False})
    """
    if not torch.is_tensor(tensor):
        tensor = torch.from_numpy(tensor)

    if is_label:
        tensor = shuffle_lr(tensor).flip(tensor.ndimension() - 1)
    else:
        tensor = tensor.flip(tensor.ndimension() - 1)

    return tensor


def load_file_from_url(url, model_dir=None, progress=True, check_hash=False, file_name=None):
    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    return cached_file
