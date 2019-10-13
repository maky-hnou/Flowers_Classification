"""Apply image processing techniques set in utils module."""

import glob

import cv2
from natsort import natsorted, ns
from utils import (image_to_npy, random_horizontal_flip, random_resized_crop,
                   random_rotation, rename_files)

if (__name__ == '__main__'):
    data_folders = ['flower_data/train/',
                    'flower_data/test/', 'flower_data/valid/']
    for folder in data_folders:
        for image in natsorted(glob.glob(folder + '**/*'),
                               alg=ns.IGNORECASE):
            try:
                print(image)
                img = cv2.imread(image)
                rotated_img = random_rotation(img)
                resized_img = random_resized_crop(rotated_img)
                flipped_img = random_horizontal_flip(resized_img)
                cv2.imwrite(image, flipped_img)
            except Exception as e:
                print(e)
                continue

        # Rename data/files
        rename_files(folder)

        # Convert data/files to numpy array
        image_to_npy(folder.split('/')[-2], folder, (224, 224))
