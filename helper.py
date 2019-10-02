"""Class Helper."""

import math
import random

import cv2
import numpy as np


class Helper:
    """Short summary.

    Parameters
    ----------
    image : type
        Description of parameter `image`.
    angle : type
        Description of parameter `angle`.
    size : type
        Description of parameter `size`.
    mean : type
        Description of parameter `mean`.
    std : type
        Description of parameter `std`.

    Attributes
    ----------
    image
    angle
    size
    mean
    std

    """

    def __init__(self, image, size, mean, std):
        """Short summary.

        Parameters
        ----------
        image : type
            Description of parameter `image`.
        size : type
            Description of parameter `size`.
        mean : type
            Description of parameter `mean`.
        std : type
            Description of parameter `std`.

        Returns
        -------
        type
            Description of returned object.

        """
        self.image = image
        self.size = size
        self.mean = mean
        self.std = std

    def random_rotation(self):
        """Short summary.

        Returns
        -------
        type
            Description of returned object.

        """
        h, w = self.image.shape[:2]
        img_c = (w / 2, h / 2)
        angle = random.randrange(30)
        rot = cv2.getRotationMatrix2D(img_c, angle, 1)

        rad = math.radians(angle)
        sin = math.sin(rad)
        cos = math.cos(rad)
        b_w = int((h * abs(sin)) + (w * abs(cos)))
        b_h = int((h * abs(cos)) + (w * abs(sin)))

        rot[0, 2] += ((b_w / 2) - img_c[0])
        rot[1, 2] += ((b_h / 2) - img_c[1])

        outImg = cv2.warpAffine(
            self.image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
        return outImg

    def random_resized_crop(self):
        """Short summary.

        Returns
        -------
        type
            Description of returned object.

        """
        resized = cv2.resize(self.image, self.size)
        return resized

    def random_horizontal_flip(self):
        """Flip an input array.

        Returns
        -------
        numpy ndarray
            The flipped array.

        """
        if (random.random() > 0.5):
            flipped = np.flipud(self.image)
        return flipped

    def normalize(self):
        """Normalize an input array.

        Returns
        -------
        numpy ndarray
            The normalized array.

        """
        normalized = self.image - self.mean
        normalized = normalized / self.std
        return normalized
