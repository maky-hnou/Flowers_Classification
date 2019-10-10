"""Apply image processing techniques set in utils module."""

import glob

import cv2
from helper import random_horizontal_flip, random_resized_crop, random_rotation
from natsort import natsorted, ns


def preprocess_img(image_path):
    """Apply preprocessing techniques on input image.

    Parameters
    ----------
    image_path : str
        The path of the image to be processed.

    Returns
    -------
    None

    """
    img = cv2.imread(image_path)
    rotated_img = random_rotation(img)
    resized_img = random_resized_crop(rotated_img)
    flipped_img = random_horizontal_flip(resized_img)
    cv2.imwrite(image_path, flipped_img)


for img in natsorted(glob.glob('flower_data/train/**/*'), alg=ns.IGNORECASE):
    try:
        preprocess_img(img)
    except AttributeError:
        continue
