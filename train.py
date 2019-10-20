"""Train the built VGG_16 model."""
import numpy as np
import tensorFlow as tf
from vgg_16 import VGG16


class Train:
    def __init__(self, train_x, train_y, valid_x=None, valid_y=None,
                 batch_size=10, learning_rate=0.01, num_epochs=1,
                 save_model=False):
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.format_size = [224, 224]
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.save_model = save_model

    def train_model(self):
        assert len(self.train_x.shape) == 4
        [num_images, img_height, img_width, num_channels] = self.train_x.shape
        num_classes = self.train_y.shape[-1]
        num_steps = int(np.ceil(num_images / float(self.batch_size)))

        # build the graph and define objective function
        graph = tf.Graph()
        with graph.as_default():
            # build graph
            train_maps_raw = tf.placeholder(
                tf.float32, [None, img_height, img_width, num_channels])
            train_maps = tf.image.resize_images(
                train_maps_raw, self.format_size[0], self.format_size[1])
            train_labels = tf.placeholder(tf.float32, [None, num_classes])
            # logits, parameters = vgg16(train_maps, num_classes)
            logits = VGG16(train_maps, num_classes,
                           isTrain=True, keep_prob=0.6)

            # loss function
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits, train_labels)
            loss = tf.reduce_mean(cross_entropy)

            # optimizer with decayed learning rate
            global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate, global_step,
                num_steps*self.num_epochs, 0.1, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(
                self.learning_rate).minimize(loss, global_step=global_step)

            # prediction for the training data
            train_prediction = tf.nn.softmax(logits)

        # train the graph
        with tf.Session(graph=graph) as session:
            # saver to save the trained model
            saver = tf.train.Saver()
            session.run(tf.initialize_all_variables())

            for epoch in range(self.num_epochs):
                for step in range(num_steps):
                    offset = (
                        step * self.batch_size) % (num_images -
                                                   self.batch_size)
                    batch_data = self.train_x[offset:(
                        offset + self.batch_size), :, :, :]
                    batch_labels = self.train_y[offset:(
                        offset + self.batch_size), :]
                    feed_dict = {train_maps_raw: batch_data,
                                 train_labels: batch_labels}
                    _, l, predictions = session.run(
                        [optimizer, loss, train_prediction],
                        feed_dict=feed_dict)

                print('Epoch %2d/%2d:\n\tTrain Loss = %.2f\t Accuracy = %.2f%%'
                      % (epoch+1, self.num_epochs, l,
                         self.accuracy(predictions, batch_labels)))
                if self.valid_x is not None and self.valid_y is not None:
                    feed_dict = {train_maps_raw: self.valid_x,
                                 train_labels: self.valid_y}
                    l, predictions = session.run(
                        [loss, train_prediction], feed_dict=feed_dict)
                    print('\tValid Loss = %.2f\t Accuracy = %.2f%%' %
                          (l, self.accuracy(predictions, self.valid_y)))

            # Save the variables to disk
            if self.save_model:
                save_path = saver.save(session, 'model.tensorflow')
                print('The model has been saved to ' + save_path)
            session.close()

    def accuracy(self, predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) ==
                               np.argmax(labels, 1)) / predictions.shape[0])
