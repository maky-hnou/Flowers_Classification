"""Test the model."""
import cv2
import tensorflow as tf
from utils import get_categories_names
from vgg_16 import BuildVGG16


class Test:
    """Test the VGG16 model.

    Parameters
    ----------
    image : str
        The path of the image to be classified.
    graph_path: str
        The path of the saved model.

    Attributes
    ----------
    graph : tf Graph
        A TensorFlow computation, represented as a dataflow graph.
    json_file_path : str
        The path to the json file containg the classes names.
    image

    """

    def __init__(self, image, graph_path):
        """__init__ Constructor.

        Parameters
        ----------
        image : numpy ndarray
            The input image to be classified.

        Returns
        -------
        None

        """
        self.image = image
        self.graph = tf.Grapgh()
        self.json_file_path = 'categories_names.json'
        self.graph_path = graph_path

    def load_image(self):
        """Load the input image.

        Returns
        -------
        numpy ndarray
            The resized input image.

        """
        img = cv2.imread(self.image)
        resized_img = cv2.resize(img, (224, 224))
        return resized_img

    def build_graph(self):
        """Build the graph skeleton.

        Returns
        -------
        output: tf tensors
            The built model's output.
        values: tf tensor
            The values of the top K elements of the built model's output.
        indices: tf tensor
            The indices of the top K elements of the built model's output.
        input_maps: tf tensor
            A handle used to feed values to the model.
        k: int
            The number of top elements

        """
        with self.graph.as_default():
            input_maps = tf.placeholder(tf.float32, [None, 224, 224, 3])
            # zero mean of input
            mean = tf.constant([103.939, 116.779, 123.68],
                               dtype=tf.float32, shape=[1, 1, 1, 3])

            model = BuildVGG16(input_maps - mean, return_all=True)
            output = model.build_model()
            softmax = tf.nn.softmax(output[-3])
            # Finds values and indices of the k largest entries
            k = 3
            values, indices = tf.nn.top_k(softmax, k)
        return output, values, indices, input_maps, k

    def run_graph(self):
        """Run the restored model.

        Returns
        -------
        None

        """
        with tf.Session(graph=self.graph) as sess:
            saver = tf.train.Saver()
            print('Restoring VGG16 model parameters ...')
            saver.restore(sess, self.graph_path)
            output, values, indices, input_maps, k = self.build_graph()
            # Load the image
            image = self.load_image(self.image)

            # Testing on the sample image
            [prob, ind, out] = sess.run([values, indices, output],
                                        feed_dict={input_maps: [image]})
            prob = prob[0]
            ind = ind[0]
            class_names = get_categories_names(self.json_file_path)
            class_names = list(class_names.values())
            print('\nClassification Result:')
            for i in range(k):
                print('\tCategory Name: %s \n\tProbability: %.2f%%\n' %
                      (class_names[ind[i]], prob[i]*100))
            sess.close()
