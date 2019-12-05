"""Build the VGG16 CNN."""
import time

import tensorflow as tf


class VGG16:
    """Build the VGG16 model.

    Parameters
    ----------
    input_shape : list
        The shape of the input data.
    num_classes : int
        The number of classes.
    isTraining : bool
        True when traing the model, False otherwhise.
    keep_prob : float
        The dropout probability.
    return_all : bool
        True to return all the outputs, False otherwhise.

    Attributes
    ----------
    input_shape
    num_classes
    isTraining
    keep_prob
    return_all

    """

    def __init__(self, input_shape, num_classes=102, isTraining=False,
                 keep_prob=1.0, return_all=False):
        """__init__ Constructor.

        Parameters
        ----------
        input_shape : list
            The shape of the input data.
        num_classes : int
            The number of classes.
        isTraining : bool
            True when traing the model, False otherwhise.
        keep_prob : float
            The dropout probability.
        return_all : bool
            True to return all the outputs, False otherwhise.

        Returns
        -------
        None

        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.isTraining = isTraining
        self.keep_prob = keep_prob
        self.return_all = return_all

    def build_model(self):
        """Build the model layers.

        Returns
        -------
        tf tensors
            The output of the model.

        """
        start_time = time.time()
        print("building model started")

        # ___Layer 1___
        # Build Convolution layer + Relu and load weights
        output1_1, kernel1_1, bias1_1 = self.convolution_layer(
            'conv1_1', self.input_shape, 64)
        # Build Convolution layer + Relu and load weights
        output1_2, kernel1_2, bias1_2 = self.convolution_layer(
            'conv1_2', output1_1, 64)
        # Build Max Pooling layer
        output1_3 = self.max_pooling_layer('pool1', output1_2)

        # ___Layer 2___
        # Build Convolution layer + Relu and load weights
        output2_1, kernel2_1, bias2_1 = self.convolution_layer(
            'conv2_1', output1_3, 128)
        # Build Convolution layer + Relu and load weights
        output2_2, kernel2_2, bias2_2 = self.convolution_layer(
            'conv2_2', output2_1, 128)
        # Build Max Pooling layer
        output2_3 = self.max_pooling_layer('pool2', output2_2)

        # ___Layer 3___
        # Build Convolution layer + Relu and load weights
        output3_1, kernel3_1, bias3_1 = self.convolution_layer(
            'conv3_1', output2_3, 256)
        # Build Convolution layer + Relu and load weights
        output3_2, kernel3_2, bias3_2 = self.convolution_layer(
            'conv3_2', output3_1, 256)
        # Build Convolution layer + Relu and load weights
        output3_3, kernel3_3, bias3_3 = self.convolution_layer(
            'conv3_3', output3_2, 256)
        # Build Max Pooling layer
        output3_4 = self.max_pooling_layer('pool3', output3_3)

        # ___Layer 4___
        # Build Convolution layer + Relu and load weights
        output4_1, kernel4_1, bias4_1 = self.convolution_layer(
            'conv4_1', output3_4, 512)
        # Build Convolution layer + Relu and load weights
        output4_2, kernel4_2, bias4_2 = self.convolution_layer(
            'conv4_2', output4_1, 512)
        # Build Convolution layer + Relu and load weights
        output4_3, kernel4_3, bias4_3 = self.convolution_layer(
            'conv4_3', output4_2, 512)
        # Build Max Pooling layer
        output4_4 = self.max_pooling_layer('pool4', output4_3)

        # ___Layer 5___
        # Build Convolution layer + Relu and load weights
        output5_1, kernel5_1, bias5_1 = self.convolution_layer(
            'conv5_1', output4_4, 512)
        # Build Convolution layer + Relu and load weights
        output5_2, kernel5_2, bias5_2 = self.convolution_layer(
            'conv5_2', output5_1, 512)
        # Build Convolution layer + Relu and load weights
        output5_3, kernel5_3, bias5_3 = self.convolution_layer(
            'conv5_3', output5_2, 512)
        # Build Max Pooling layer
        output5_4 = self.max_pooling_layer('pool5', output5_3)

        # Drop out to avoid over fitting
        if (self.isTraining):
            output5_4 = tf.nn.dropout(output5_4, keep_prob=self.keep_prob)

        # ___Fully Connected Layer 1___
        output6_1, kernel6_1, bias6_1 = self.fully_connected_layer(
            'fc6_1', output5_4, 4096)
        # ___Fully Connected Layer 2___
        output6_2, kernel6_2, bias6_2 = self.fully_connected_layer(
            'fc6_2', output6_1, 4096)

        # ___Fully Connected Layer 3___
        output6_3, kernel6_3, bias6_3 = self.fully_connected_layer(
            'fc6_3', output6_2, self.num_classes)

        # Soft Max Layer
        prob = tf.nn.softmax(output6_3, name="prob")
        print('#' * 32)
        print('softmax:', output6_3)
        print('#' * 32)

        print(('build model finished in:', (time.time() - start_time)))

        if (self.return_all):
            return output1_1, kernel1_1, bias1_1,\
                output1_2, kernel1_2, bias1_2,\
                output2_1, kernel2_1, bias2_1,\
                output2_2, kernel2_2, bias2_2,\
                output3_1, kernel3_1, bias3_1,\
                output3_2, kernel3_2, bias3_2,\
                output3_3, kernel3_3, bias3_3,\
                output4_1, kernel4_1, bias4_1,\
                output4_2, kernel4_2, bias4_2,\
                output4_3, kernel4_3, bias4_3,\
                output5_1, kernel5_1, bias5_1,\
                output5_2, kernel5_2, bias5_2,\
                output5_3, kernel5_3, bias5_3,\
                output6_1, kernel6_1, bias6_1,\
                output6_2, kernel6_2, bias6_2,\
                output6_3, kernel6_3, bias6_3

        else:
            # return output6_3
            return list(prob)

    # construct a Convolutional Layer
    def convolution_layer(self, layer_name, input_maps, num_output_channels,
                          kernel_size=[3, 3], stride=[1, 1, 1, 1]):
        """Create a Convolutional layer.

        Parameters
        ----------
        layer_name : str
            The name of the layer.
        input_maps : tf tensor
            The input data.
        num_output_channels : int
            The number of color channels.
        kernel_size : list
            The shape of the filter.
        stride : list
            The stride of the sliding window for each dimension of input_maps.

        Returns
        -------
        tf tensors
            The outputs of the convolutional layer.

        """
        num_input_channels = input_maps.get_shape()[-1].value
        with tf.name_scope(layer_name) as scope:
            kernel = tf.get_variable(
                scope+'W', shape=[kernel_size[0], kernel_size[1],
                                  num_input_channels, num_output_channels],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer_conv2d())
            convolution = tf.nn.conv2d(
                input_maps, kernel, stride, padding='SAME')
            bias = tf.Variable(
                tf.constant(
                    0.0, shape=[num_output_channels], dtype=tf.float32),
                trainable=True, name='b')
            output = tf.nn.relu(tf.nn.bias_add(convolution, bias), name=scope)
            return output, kernel, bias

    # Construct a Max Pooling Layer
    def max_pooling_layer(self, layer_name, input_maps,
                          kernel_size=[2, 2], stride=[1, 2, 2, 1]):
        """Perform the max pooling on the input.

        Parameters
        ----------
        layer_name : str
            The name of the layer.
        input_maps : tf tensor
            The input data.
        kernel_size : list
            The shape of the filter.
        stride : list
            The stride of the sliding window for each dimension of input_maps.

        Returns
        -------
        tf tensor
            The max pooled output tensor.

        """
        output = tf.nn.max_pool(input_maps,
                                ksize=[1, kernel_size[0], kernel_size[1], 1],
                                strides=stride,
                                padding='SAME',
                                name=layer_name)
        return output

    # Construct an Average Pooling Layer

    def avg_pooling_layer(self, layer_name, input_maps,
                          kernel_size=[2, 2], stride=[1, 2, 2, 1]):
        """Perform the average pooling on the input.

        Parameters
        ----------
        layer_name : str
            The name of the layer.
        input_maps : tf tensor
            The input data.
        kernel_size : list
            The shape of the filter.
        stride : list
            The stride of the sliding window for each dimension of input_maps.

        Returns
        -------
        tf tensor
            The max pooled output tensor.

        """
        output = tf.nn.avg_pool(input_maps,
                                ksize=[1, kernel_size[0], kernel_size[1], 1],
                                strides=stride,
                                padding='SAME',
                                name=layer_name)
        return output

    # Construct a Fully Connected Layer

    def fully_connected_layer(self, layer_name, input_maps, num_output_nodes):
        """Create a fully connected layer.

        Parameters
        ----------
        layer_name : str
            The name of the layer.
        input_maps : tf tensor
            The input data.
        num_output_nodes : int
            The number of the output nodes.

        Returns
        -------
        tf tensors
            The outputs of the fully connected layer.

        """
        shape = input_maps.get_shape()
        if len(shape) == 4:
            size = shape[1].value * shape[2].value * shape[3].value
        else:
            size = shape[-1].value
        with tf.name_scope(layer_name) as scope:
            kernel = tf.get_variable(
                scope+'W', shape=[size, num_output_nodes],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.Variable(
                tf.constant(
                    0.1, shape=[num_output_nodes], dtype=tf.float32),
                trainable=True, name='b')
            flat = tf.reshape(input_maps, [-1, size])
            output = tf.nn.relu(tf.nn.bias_add(tf.matmul(flat, kernel), bias))
            return output, kernel, bias
