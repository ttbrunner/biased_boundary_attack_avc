import tensorflow.contrib.layers as layers
import tensorflow as tf
import numpy as np
import os
from timeit import default_timer

from scipy.ndimage import gaussian_filter


class DunetDenoiser:
    """
    Reimplemented the winning defense of NIPS2017:
    Liao et al., "Defense against Adversarial Attacks Using High-Level Representation Guided Denoiser"
    https://arxiv.org/abs/1712.02976

    Conclusion: Not very successful in the scenario of the 2018 competition.
    """

    def __init__(self, session=None, tf_x=None):
        self._created_session = False
        if session is None:
            session = tf.get_default_session()
            if session is None:
                session = tf.Session()
                self._created_session = True
        self._session = session

        with session.graph.as_default():
            # Norm input
            if tf_x is None:
                tf_x = tf.placeholder(tf.float32, [None, 64, 64, 3])
            self._tf_x = tf_x
            tf_x_normed = tf_x / 255.

            # Create Unet
            scope = 'denoiser'
            tf_noise_normed = build_unet(tf_x_normed, initializer=None, is_training=False, scope=scope)
            self._tf_noise = tf_noise_normed * 255.

            var_list = [v for v in tf.global_variables() if v.name.split('/')[0] == scope]
            denoiser_saver = tf.train.Saver(var_list)

        # Load checkpoint with Saver
        path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(path, 'checkpoints', 'denoiser-26-96.5881.ckpt')
        denoiser_saver.restore(session, path)

    def do_denoise(self, X_input):
        X_input = np.float32(X_input)

        time_start = default_timer()
        X_noise = self._session.run(self._tf_noise, feed_dict={self._tf_x: X_input})
        time_elapsed = default_timer() - time_start

        # print("Denoised {} images in {:.1f} ms.".format(X_noise.shape[0], time_elapsed * 1000))
        # for i in range(X_noise.shape[0]):
        #     print("Noise magnitude: {}".format(np.linalg.norm(X_noise / 255.)))

        # Idea 1: First denoise, then apply a strong blur where the noise was (directly based on noise pattern).
        # X_denoised = X_input - 1.25 * X_noise
        # X_gauss = gaussian_filter(X_denoised, sigma=0.8)
        # X_denoised = X_denoised + np.abs(X_noise / 255.) * (X_gauss - X_denoised)

        # Idea 2: Blur the noise pattern, then denoise.
        X_noise += gaussian_filter(X_noise, sigma=0.6)
        X_denoised = X_input - 1 * X_noise

        DEBUG=False
        if DEBUG:
            from matplotlib import pyplot as plt
            for i in range(len(X_input)):
                plt.imshow(np.uint8(np.round(np.clip(X_input[i], 0, 255))))
                plt.figure()
                plt.imshow(np.uint8(np.round(np.clip(X_noise[i], 0, 255))))
                plt.figure()
                plt.imshow(np.uint8(np.round(np.clip(X_denoised[i], 0, 255))))
                plt.show(block=True)

        return np.clip(X_denoised, 0., 255.)


# Code adapted from https://github.com/ankurhanda/tf-unet
def conv_bn_layer(input_tensor, kernel_size,output_channels,
                  initializer, stride=1, bn=False,
                  is_training=True, relu=True):

    # with tf.variable_scope(name) as scope:
    conv_layer = layers.conv2d(inputs=input_tensor,
                               num_outputs=output_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               activation_fn=tf.identity,
                               padding='SAME',
                               weights_initializer=initializer)
    if bn and relu:
        #How to use Batch Norm: https://github.com/martin-gorner/tensorflow-mnist-tutorial/blob/master/README_BATCHNORM.md

        #Why scale is false when using ReLU as the next activation
        #https://datascience.stackexchange.com/questions/22073/why-is-scale-parameter-on-batch-normalization-not-needed-on-relu/22127

        #Using fuse operation: https://www.tensorflow.org/performance/performance_guide#common_fused_ops
        conv_layer = layers.batch_norm(inputs=conv_layer, center=True, scale=False, is_training=is_training, fused=True)
        conv_layer = tf.nn.relu(conv_layer)

    if bn and not relu:
        conv_layer = layers.batch_norm(inputs=conv_layer, center=True, scale=True, is_training=is_training)

    # print('Conv layer {0} -> {1}'.format(input_tensor.get_shape().as_list(),conv_layer.get_shape().as_list()))
    return conv_layer


def build_unet(x, initializer, is_training, scope='denoiser'):
    with tf.variable_scope(scope) as scope:
        # First C2
        conv_layer = conv_bn_layer(x, kernel_size=(3, 3),
                                   output_channels=64, initializer=initializer,
                                   stride=1, bn=True, is_training=is_training, relu=True)
        conv_layer_enc_64 = conv_bn_layer(conv_layer, kernel_size=(3, 3),
                                          output_channels=64, initializer=initializer,
                                          stride=1, bn=True, is_training=is_training, relu=True)

        # Fist C3
        conv_layer = conv_bn_layer(conv_layer_enc_64, kernel_size=(3, 3),
                                   output_channels=128, initializer=initializer,
                                   stride=2, bn=True, is_training=is_training, relu=True)
        conv_layer = conv_bn_layer(conv_layer, kernel_size=(3, 3),
                                   output_channels=128, initializer=initializer,
                                   stride=1, bn=True, is_training=is_training, relu=True)
        conv_layer_enc_128 = conv_bn_layer(conv_layer, kernel_size=(3, 3),
                                           output_channels=128, initializer=initializer,
                                           stride=1, bn=True, is_training=is_training, relu=True)

        # 2nd C3
        conv_layer = conv_bn_layer(conv_layer_enc_128, kernel_size=(3, 3),
                                   output_channels=256, initializer=initializer,
                                   stride=2, bn=True, is_training=is_training, relu=True)
        conv_layer = conv_bn_layer(conv_layer, kernel_size=(3, 3),
                                   output_channels=256, initializer=initializer,
                                   stride=1, bn=True, is_training=is_training, relu=True)
        conv_layer_enc_256 = conv_bn_layer(conv_layer, kernel_size=(3, 3),
                                             output_channels=256, initializer=initializer,
                                             stride=1, bn=True, is_training=is_training, relu=True)

        # 3rd C3
        conv_layer = conv_bn_layer(conv_layer_enc_256, kernel_size=(3, 3),
                                   output_channels=256, initializer=initializer,
                                   stride=2, bn=True, is_training=is_training, relu=True)
        conv_layer = conv_bn_layer(conv_layer, kernel_size=(3, 3),
                                   output_channels=256, initializer=initializer,
                                   stride=1, bn=True, is_training=is_training, relu=True)
        conv_layer_dec_256 = conv_bn_layer(conv_layer, kernel_size=(3, 3),
                                           output_channels=256, initializer=initializer,
                                           stride=1, bn=True, is_training=is_training, relu=True)

        # Now backward path
        # First
        reduced_patchsize = np.multiply(conv_layer_dec_256.get_shape().as_list()[1:3], 2)
        conv_layer_dec_256 = tf.image.resize_images(conv_layer_dec_256, size=reduced_patchsize,
                                                    method=tf.image.ResizeMethod.BILINEAR)

        conv_layer = tf.concat([conv_layer_dec_256, conv_layer_enc_256], axis=3)
        conv_layer = conv_bn_layer(conv_layer, kernel_size=(3, 3),
                                   output_channels=256, initializer=initializer,
                                   stride=1, bn=True, is_training=is_training, relu=True)
        conv_layer = conv_bn_layer(conv_layer, kernel_size=(3, 3),
                                   output_channels=256, initializer=initializer,
                                   stride=1, bn=True, is_training=is_training, relu=True)
        conv_layer = conv_bn_layer(conv_layer, kernel_size=(3, 3),
                                   output_channels=256, initializer=initializer,
                                   stride=1, bn=True, is_training=is_training, relu=True)

        # Second
        reduced_patchsize = np.multiply(conv_layer.get_shape().as_list()[1:3], 2)
        conv_layer = tf.image.resize_images(conv_layer, size=reduced_patchsize,
                                            method=tf.image.ResizeMethod.BILINEAR)

        conv_layer = tf.concat([conv_layer, conv_layer_enc_128], axis=3)
        conv_layer = conv_bn_layer(conv_layer, kernel_size=(3, 3),
                                   output_channels=128, initializer=initializer,
                                   stride=1, bn=True, is_training=is_training, relu=True)
        conv_layer = conv_bn_layer(conv_layer, kernel_size=(3, 3),
                                   output_channels=128, initializer=initializer,
                                   stride=1, bn=True, is_training=is_training, relu=True)
        conv_layer = conv_bn_layer(conv_layer, kernel_size=(3, 3),
                                   output_channels=128, initializer=initializer,
                                   stride=1, bn=True, is_training=is_training, relu=True)

        # Third
        reduced_patchsize = np.multiply(conv_layer.get_shape().as_list()[1:3], 2)
        conv_layer = tf.image.resize_images(conv_layer, size=reduced_patchsize,
                                            method=tf.image.ResizeMethod.BILINEAR)

        conv_layer = tf.concat([conv_layer, conv_layer_enc_64], axis=3)
        conv_layer = conv_bn_layer(conv_layer, kernel_size=(3, 3),
                                   output_channels=64, initializer=initializer,
                                   stride=1, bn=True, is_training=is_training, relu=True)
        conv_layer = conv_bn_layer(conv_layer, kernel_size=(3, 3),
                                   output_channels=64, initializer=initializer,
                                   stride=1, bn=True, is_training=is_training, relu=True)
        conv_layer = conv_bn_layer(conv_layer, kernel_size=(3, 3),
                                   output_channels=64, initializer=initializer,
                                   stride=1, bn=True, is_training=is_training, relu=True)

        prediction = layers.conv2d(conv_layer, num_outputs=3, kernel_size=(1, 1),
                                   stride=1, padding='SAME', weights_initializer=initializer,
                                   activation_fn=tf.identity)

        return prediction
