
"""
CV operations

Authors: Jorn Tuyls
"""

import cv2
import numpy as np

def crop(height, width, channels):
    # type: (List[str/int], List[str/int], List[str,int]) -> Function
    """
    Return a wrapper function that takes in an image and crops it according to 
    provided height, width and channel boundaries.
    """

    assert(len(height) == 2 and len(width) == 2 and len(channels) == 2)
    print(height, width, channels)
    
    start_h, end_h = int(height[0]), int(height[1])
    start_w, end_w = int(width[0]), int(width[1])
    start_c, end_c = int(channels[0]), int(channels[1])

    def _crop(img):
        # img should be in HWC layout
        return img[start_h:end_h, start_w:end_w, start_c:end_c]

    return _crop

def normalize(means, stdevs):
    # type: (List[str/int/float], List[str/int/float]) -> Function
    """
    Return a wrapper function to normalize an image according to provided 
    means and standard deviations.
    """
    assert(len(means) == len(stdevs))
    
    means = [float(mean) for mean in means]
    stdevs = [float(stdev) for stdev in stdevs]

    def _normalize(img):
        # img should be in HWC layout
        assert(img.shape[2] == len(means))
        return (img - means) / stdevs

    return _normalize

def resize(size):    
    # type: (List[str/int]) -> Function
    """
    Return a wrapper function to resize an image to provided size.

    Arguments
    ---------
    size: List[str/int]
        the new size as a width, height tuple
    """
    assert(len(size) == 2)

    size = [int(dim) if dim not in ['?', 'None', None] else None for dim in size]
    assert(size != [None, None])

    def _resize(img):
        # !! img should be in HWC format
        if img.dtype not in ['float32']:
            raise ValueError("OpenCV resize operator expects imput array"\
                " to have float32 data type but got: {}".format(img.dtype))

        if size[0] is None:
            height_aspect_ratio = size[1] / float(img.shape[0])
            size[0] = int(img.shape[1] * height_aspect_ratio)
        elif size[1] is None:
            width_aspect_ratio = size[0] / float(img.shape[1])
            size[1] = int(img.shape[0] * width_aspect_ratio)

        return cv2.resize(img, tuple(size))

    return _resize

def scale(scale):
    # type: (str/int/float) -> Function
    """
    Return a wrapper function that takes in an image and scales it according to 
    the provided scale argument.
    """

    def _scale(img):
        return img * float(scale)

    return _scale

def subtract(values):
    # type: (List[str/int/float]) -> Function
    """
    Return a wrapper function that takes in an image and subtracts the provided
    values
    """

    def _subtract(img):
        return img - values

    return _subtract

def transpose(axes):
    # type: (List[str/[int]) -> Function
    """
    Return a wrapper function takes in an image and transposes it according to 
    the provided axes argument.
    """

    axes = [int(axis) for axis in axes]

    def _transpose(img):
        return np.transpose(img, axes=axes)

    return _transpose