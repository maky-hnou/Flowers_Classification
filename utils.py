"""Utils functions."""

import math
import random

import cv2
import numpy as np


def random_rotation(image):
    """Apply random rotation to an input image.

    Returns
    -------
    numpy ndarray
        The rotated image.

    """
    height, width = image.shape[:2]
    img_dims = (width / 2, height / 2)
    angle = random.randrange(30)
    rotated_img = cv2.getRotationMatrix2D(img_dims, angle, 1)

    rad = math.radians(angle)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((height * abs(sin)) + (width * abs(cos)))
    b_h = int((height * abs(cos)) + (width * abs(sin)))

    rotated_img[0, 2] += ((b_w / 2) - img_dims[0])
    rotated_img[1, 2] += ((b_h / 2) - img_dims[1])

    wrapped_img = cv2.warpAffine(image, rotated_img, (b_w, b_h),
                                 flags=cv2.INTER_LINEAR)
    return wrapped_img


def random_resized_crop(image, size=(224, 224)):
    """Resize an inpu image.

    Returns
    -------
    numpy ndarray
        The resized image.

    """
    resized_img = cv2.resize(image, size)
    return resized_img


def random_horizontal_flip(image):
    """Flip an input array.

    Returns
    -------
    numpy ndarray
        The flipped array (image).

    """
    if (random.random() > 0.5):
        flipped_img = np.flipud(image)
        return flipped_img
    else:
        return image


def normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Normalize an input array.

    Returns
    -------
    numpy ndarray
        The normalized array.

    """
    normalized_img = image - mean
    normalized_img = normalized_img / std
    return normalized_img.astype(np.uint8)
