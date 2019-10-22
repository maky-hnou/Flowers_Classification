"""Train the CNN."""
import numpy as np
from train import Train


def load_data(path):
    """Load data saved into .npy file.

    Parameters
    ----------
    path : str
        The path to the .npy file.

    Returns
    -------
    numpy ndarrays
        The data and its labels.

    """
    data = np.load(path)
    images = np.array([i[0] for i in data])
    labels = np.array([i[1] for i in data])
    return images, labels


if (__name__ == '__main__'):
    train_data_path = 'train_data_gray.npy'
    valid_data_path = ''
    print('Loading dataset ...')
    train_imgs, train_labels = load_data(train_data_path)
    valid_imgs, valid_labels = load_data(valid_data_path)
    train = Train(train_x=train_imgs, train_y=train_labels,
                  valid_x=valid_imgs, valid_y=valid_labels, batch_size=10,
                  learning_rate=0.01, num_epochs=200, save_model=True)
    train.train_model()