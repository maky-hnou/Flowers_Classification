"""Class Helper."""

import math
import random

import cv2
import numpy as np


class Helper:
    """Helper class.

    Parameters
    ----------
    image : numpy ndarray
        The input image.

    Attributes
    ----------
    size : tuple
        The new size of the image.
    mean : list
        The list of means.
    std : type
        The list of standard deviations.

    """

    def __init__(self, image):
        """__init__ Constructor.

        Parameters
        ----------
        image : numpy ndarray
            The input image.

        Returns
        -------
        None

        """
        self.image = image
        self.size = (224, 224)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def random_rotation(self):
        """Apply random rotation to an input image.

        Returns
        -------
        numpy ndarray
            The rotated image.

        """
        height, width = self.image.shape[:2]
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

        out_img = cv2.warpAffine(
            self.image, rotated_img, (b_w, b_h), flags=cv2.INTER_LINEAR)
        return out_img

    def random_resized_crop(self):
        """Resize an inpu image.

        Returns
        -------
        numpy ndarray
            The resized image.

        """
        rotated_img = self.random_rotation()
        resized_img = cv2.resize(rotated_img, self.size)
        return resized_img

    def random_horizontal_flip(self):
        """Flip an input array.

        Returns
        -------
        numpy ndarray
            The flipped array (image).

        """
        resized_img = self.random_resized_crop()
        if (random.random() > 0.5):
            flipped_img = np.flipud(self.image)
            return flipped_img
        else:
            return resized_img

    def normalize(self):
        """Normalize an input array.

        Returns
        -------
        numpy ndarray
            The normalized array.

        """
        flipped_img = self.random_horizontal_flip()
        normalized_img = flipped_img - self.mean
        normalized_img = normalized_img / self.std
        return normalized_img.astype(np.uint8)
