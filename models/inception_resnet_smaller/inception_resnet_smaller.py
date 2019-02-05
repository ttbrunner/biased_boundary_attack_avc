from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow import layers


class InceptionResnetSmaller:
    def __init__(self, n_classes, is_training=False):
        self.n_classes = n_classes
        self.is_training = is_training

    def build_network(self, img_placeholder):
        x = img_placeholder
        x = self.basic_conv2d(x, 32, kernel_size=3,stride=2, padding="same",
                              namescope="conv2d_1a", use_bias=False)
        x = self.basic_conv2d(x, 32, kernel_size=3, stride=1, padding="same",
                              namescope="conv2d_2a", use_bias=False)
        x = self.basic_conv2d(x, 64, kernel_size=3, stride=1, padding="same",
                              namescope="conv2d_2b", use_bias=False)
        x = self.basic_conv2d(x, 80, kernel_size=1, stride=1, padding="same",
                              namescope="conv2d_3b", use_bias=False)
        x = self.basic_conv2d(x, 192, kernel_size=3, stride=1, padding="same",
                              namescope="conv2d_4a", use_bias=False)

        x = self.mixed_5b(x)
        x = self.mixed_6a(x)
        x = self.mixed_7a(x)
        x = self.block8(x)

        x = self.basic_conv2d(x, 1536, kernel_size=1,stride=1, padding="same",
                              namescope="conv2d_7b", use_bias=False)
        x = layers.average_pooling2d(x, 8, strides=8, padding="valid")
        x = layers.flatten(x)
        logits = layers.dense(x, self.n_classes, name="last_linear")
        probs = tf.nn.softmax(logits)
        return logits, probs

    def block8(self, x):
        initial = x
        with tf.variable_scope("block8"):
            with tf.variable_scope("branch0"):
                x0 = self.basic_conv2d(x, 192, kernel_size=1, stride=1, padding="same",
                                       namescope="conv1", use_bias=False)
            with tf.variable_scope("branch1"):
                x1 = self.basic_conv2d(x, 192, kernel_size=1, stride=1, padding="same",
                                       namescope="conv1", use_bias=False)
                x1 = self.basic_conv2d(x1, 224, kernel_size=(1, 3), stride=1, padding="same",
                                       namescope="conv2", use_bias=False)
                x1 = self.basic_conv2d(x1, 256, kernel_size=(3, 1), stride=1, padding="same",
                                       namescope="conv3", use_bias=False)
            x = tf.concat([x0, x1], axis=-1)
            x = layers.conv2d(x, filters=2080,
                              kernel_size=1,
                              padding="same",
                              strides=1,
                              use_bias=True
                              )
            x = initial + x
            x = layers.dropout(x, noise_shape=[None, 1, 1, None])

        return x

    def mixed_7a(self, x):
        with tf.variable_scope("mixed_7a"):
            with tf.variable_scope("branch0"):
                x0 = self.basic_conv2d(x, 256, kernel_size=1, stride=1, padding="same",
                                       namescope="conv1", use_bias=False)
                x0 = self.basic_conv2d(x0, 384, kernel_size=3, stride=2, padding="same",
                                       namescope="conv2", use_bias=False)
            with tf.variable_scope("branch1"):
                x1 = self.basic_conv2d(x, 256, kernel_size=1, stride=1, padding="same",
                                       namescope="conv1", use_bias=False)
                x1 = self.basic_conv2d(x1, 288, kernel_size=3, stride=2, padding="same",
                                       namescope="conv2", use_bias=False)
            with tf.variable_scope("branch2"):
                x2 = self.basic_conv2d(x, 256, kernel_size=1, stride=1, padding="same",
                                       namescope="conv1", use_bias=False)
                x2 = self.basic_conv2d(x2, 288, kernel_size=3, stride=1, padding="same",
                                       namescope="conv2", use_bias=False)
                x2 = self.basic_conv2d(x2, 320, kernel_size=3, stride=2, padding="same",
                                       namescope="conv3", use_bias=False)
            with tf.variable_scope("branch3"):
                x3 = layers.max_pooling2d(x, 3, strides=2, padding="same")
            x = tf.concat([x0, x1, x2, x3], axis=-1)
            x = layers.dropout(x, noise_shape=[None, 1, 1, None])
        return x

    def mixed_6a(self, x):
        with tf.variable_scope("mixed_6a"):
            with tf.variable_scope("branch0"):
                x0 = self.basic_conv2d(x, 384, kernel_size=3, stride=2, padding="same",
                                       namescope="conv1", use_bias=False)
            with tf.variable_scope("branch1"):
                x1 = self.basic_conv2d(x, 256, kernel_size=1, stride=1, padding="same",
                                       namescope="conv1", use_bias=False)
                x1 = self.basic_conv2d(x1, 256, kernel_size=3, stride=1, padding="same",
                                       namescope="conv2", use_bias=False)
                x1 = self.basic_conv2d(x1, 384, kernel_size=3, stride=2, padding="same",
                                       namescope="conv3", use_bias=False)
            with tf.variable_scope("branch2"):
                x2 = layers.max_pooling2d(x, 3, strides=2, padding="same")
            x = tf.concat([x0, x1, x2], axis=-1)
            x = layers.dropout(x, noise_shape=[None, 1, 1, None])
        return x

    def mixed_5b(self, x):
        with tf.variable_scope("mixed_5b"):
            with tf.variable_scope("branch0"):
                x0 = self.basic_conv2d(x, 96, kernel_size=1, stride=1, padding="same", namescope="conv1", use_bias=False)
            with tf.variable_scope("branch1"):
                x1 = self.basic_conv2d(x, 48, kernel_size=1, stride=1, padding="same", namescope="conv1", use_bias=False)
                x1 = self.basic_conv2d(x1, 64, kernel_size=5, stride=1, padding="same", namescope="conv2", use_bias=False)
            with tf.variable_scope("branch2"):
                x2 = self.basic_conv2d(x, 64, kernel_size=1, stride=1, padding="same", namescope="conv1", use_bias=False)
                x2 = self.basic_conv2d(x2, 96, kernel_size=3, stride=1, padding="same", namescope="conv2", use_bias=False)
                x2 = self.basic_conv2d(x2, 96, kernel_size=3, stride=1, padding="same", namescope="conv3", use_bias=False)
            with tf.variable_scope("branch3"):
                x3 = layers.average_pooling2d(x, 3, strides=1, padding="same")
                x3 = self.basic_conv2d(x3, 64, kernel_size=1, stride=1, padding="same", namescope="conv1", use_bias=False)
            x = tf.concat([x0, x1, x2, x3], axis=-1)
        x = layers.dropout(x, noise_shape=[None, 1, 1, None])
        return x

    def basic_conv2d(self, x, filters, kernel_size, stride, padding, namescope,
                     use_bias=True):
        with tf.variable_scope(namescope):
            x = layers.conv2d(x, filters=filters,
                              kernel_size=kernel_size,
                              padding=padding,
                              strides=stride,
                              use_bias=use_bias
                              )
            x = layers.batch_normalization(x, epsilon=0.001,
                                           momentum=0.1,
                                           training=self.is_training)
                                           # affine=True)  !TODO!
            x = tf.nn.relu(x)
        return x
