"""Utils functions."""

import glob
import math
import os
import random
import re

import cv2
import numpy as np
from natsort import natsorted, ns
from tqdm import tqdm


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


def get_folder_names(json_file_path):
    """Get folder names and indexes from a json file and put them into a dict.

    Parameters
    ----------
    json_file_path : str
        The path of the json file.

    Returns
    -------
    dict
        A dictionary containing folders names and indexes.

    """
    folder_names_dict = {}
    with open(json_file_path, 'r') as foo:
        for line in foo.readlines():
            try:
                folder_index = re.search(r'\d+', line).group()
                folder_name = re.search(r': "(.*)",', line).group(1)
                folder_names_dict[folder_index] = folder_name
            except AttributeError:
                continue
    return folder_names_dict


def rename_folders(path, json_file_path):
    """Rename folders.

    Parameters
    ----------
    path : str
        The path of the parent dir.
    json_file_path : str
        The path of the json file.

    Returns
    -------
    None

    """
    names_dict = get_folder_names(json_file_path)
    for key, value in names_dict.items():
        try:
            os.rename(os.path.join(path, key), os.path.join(path, value))
        except OSError:
            print(key, value)
            continue


def rename_files(path):
    """Rename files in subdirectores of a given parent directory.

    Parameters
    ----------
    path : str
        The parent directory path.

    Returns
    -------
    None

    """
    for folder in natsorted(glob.glob(path + '/*'), alg=ns.IGNORECASE):
        i = 1
        for img in natsorted(glob.glob(folder + '/*'), alg=ns.IGNORECASE):
            path_parts = img.split('/')
            class_folder_path = path_parts[:-1]
            class_number = path_parts[-2]
            os.rename(img, '/'.join(class_folder_path)
                      + '/{}_{}.jpg'.format(class_number, i))
            i += 1


def label_images(filename, path, json_file_path):
    """Return the label of a given image existing a given directory.

    Parameters
    ----------
    filename : str
        The filename path.
    path : str
        The directory path.
    json_file_path : str
        The path of the json file.

    Returns
    -------
    list
        The label of the given image.

    """
    categories_dict = get_folder_names(json_file_path)
    classes_number = len(glob.glob(path + '/*'))
    class_labels = np.identity(classes_number, dtype=int)
    class_label = list(categories_dict.keys())[
        list(categories_dict.values()).index(filename.split('/')[-2])]
    return class_labels[int(class_label) - 1]


def image_to_npy(filename, path, img_size, json_file_path):
    """Create numpy array from an input image.

    Parameters
    ----------
    filename : str
        The name of the array of images.
    path : str
        The path of the directory where the data is.
    img_size : tuple
        The new size of the image.
    json_file_path : str
        The path of the json file.

    Returns
    -------
    None

    """
    data = []
    for img in tqdm(natsorted(glob.glob(path + '/**/*'), alg=ns.IGNORECASE)):
        label = label_images(img, path, json_file_path)
        img = cv2.imread(img, 1)
        img = cv2.resize(img, img_size)
        data.append([np.array(img), label])
    random.shuffle(data)
    np.save('{}_data_gray.npy'.format(filename), data)
