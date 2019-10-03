"""Build the VGG16 model."""
import inspect
import os
import time

import numpy as np
import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]


class VGG16:
    """VGG16 CNN.

    Parameters
    ----------
    vgg16_npy_path : type
        Description of parameter `vgg16_npy_path`.

    Attributes
    ----------
    data_dict : type
        Description of attribute `data_dict`.

    """

    def __init__(self, vgg16_npy_path=None):
        """Short summary.

        Parameters
        ----------
        vgg16_npy_path : type
            Description of parameter `vgg16_npy_path`.

        Returns
        -------
        type
            Description of returned object.

        """
        if (vgg16_npy_path is None):
            path = inspect.getfile(VGG16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print(path)

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def build(self, bgr):
        """Load variable from npy to build the VGG.

        Parameters
        ----------
        rgb : numpy ndarray
            image [batch, height, width, 3] values scaled [0, 1].

        Returns
        -------
        type
            Description of returned object.

        """
        start_time = time.time()
        print("build model started")

        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        # ___Layer 1___
        # Build Convolution layer + Relu and load weights
        self.conv1_1 = self.conv_layer(bgr, "conv1_1")

        # Build Convolution layer + Relu and load weights
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")

        # Build Max Pooling layer
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        # ___Layer 2___
        # Build Convolution layer + Relu and load weights
        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")

        # Build Convolution layer + Relu and load weights
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")

        # Build Max Pooling layer
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        # ___Layer 3___
        # Build Convolution layer + Relu and load weights
        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")

        # Build Convolution layer + Relu and load weights
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")

        # Build Convolution layer + Relu and load weights
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")

        # Build Max Pooling layer
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        # ___Layer 4___
        # Build Convolution layer + Relu and load weights
        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")

        # Build Convolution layer + Relu and load weights
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")

        # Build Convolution layer + Relu and load weights
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")

        # Build Max Pooling layer
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        # ___Layer 5___
        # Build Convolution layer + Relu and load weights
        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")

        # Build Convolution layer + Relu and load weights
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")

        # Build Convolution layer + Relu and load weights
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")

        # Build Max Pooling layer
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        # ___Fully Connected Layer 1___
        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        # Fully Connected Layer 2___
        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        # Fully Connected Layer 3___
        self.fc8 = self.fc_layer(self.relu7, "fc8")

        # Soft Max Layer
        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))

    def avg_pool(self, bottom, name):
        """Short summary.

        Parameters
        ----------
        bottom : type
            Description of parameter `bottom`.
        name : type
            Description of parameter `name`.

        Returns
        -------
        type
            Description of returned object.

        """
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME',
                              name=name)

    def max_pool(self, bottom, name):
        """Short summary.

        Parameters
        ----------
        bottom : type
            Description of parameter `bottom`.
        name : type
            Description of parameter `name`.

        Returns
        -------
        type
            Description of returned object.

        """
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        """Short summary.

        Parameters
        ----------
        bottom : type
            Description of parameter `bottom`.
        name : type
            Description of parameter `name`.

        Returns
        -------
        type
            Description of returned object.

        """
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        """Short summary.

        Parameters
        ----------
        bottom : type
            Description of parameter `bottom`.
        name : type
            Description of parameter `name`.

        Returns
        -------
        type
            Description of returned object.

        """
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        """Short summary.

        Parameters
        ----------
        name : type
            Description of parameter `name`.

        Returns
        -------
        type
            Description of returned object.

        """
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        """Short summary.

        Parameters
        ----------
        name : type
            Description of parameter `name`.

        Returns
        -------
        type
            Description of returned object.

        """
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        """Short summary.

        Parameters
        ----------
        name : type
            Description of parameter `name`.

        Returns
        -------
        type
            Description of returned object.

        """
        return tf.constant(self.data_dict[name][0], name="weights")
