"""
This is an InceptionResnetv2 that we trained ourselves - the net is a bit smaller than the usual InceptionResnetv2.
We did a little Adversarial Training on it, but without much effect (it's far inferior to the ResNet50 baseline).
"""

import tensorflow as tf
import os
from foolbox.models import TensorFlowModel

from .inception_resnet_smaller import InceptionResnetSmaller


def _create_model(graph=None, x_input=None):
    if graph is None:
        graph = tf.Graph()

    with graph.as_default():

        if x_input is None:
            x_input = tf.placeholder(tf.float32, (None, 64, 64, 3))

        # We did some experiments with HSV-transforming the input to combat transferability. Contact us if you want to know more.
        input_format = "rgb"
        if input_format == "rgb":
            images = x_input            # Not normed... at least for our old checkpoints
        elif input_format == "yuv":
            images = tf.image.rgb_to_yuv(x_input / 255.)
        elif input_format == "hsv":
            images = tf.image.rgb_to_hsv(x_input / 255.)
        else:
            raise ValueError('Unknown image format "{}"!'.format(input_format))

        scope_name = "in_res_adversarial"
        #scope_name = "in_res_adversarial_yuv"
        #scope_name = "in_res_adversarial_hsv"
        vars_before = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)
        with tf.variable_scope(scope_name):
            logits, softmax = InceptionResnetSmaller(200, is_training=False).build_network(images)
        vars_after = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)
        vars_new = set(vars_after).difference(vars_before)

        with tf.variable_scope('utilities'):
            saver = tf.train.Saver(list(vars_new))

    return graph, saver, x_input, logits


def create_ir_model(sess=None, x_input=None, foolbox=True):

    # Allow to reuse a session and put the model on top of an existing input
    assert (sess is not None) == (x_input is not None)

    if sess is not None:
        graph, saver, images, logits = _create_model(sess.graph, x_input)
    else:
        graph, saver, images, logits = _create_model(None, None)
        sess = tf.Session(graph=graph)

    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, 'checkpoints', 'bnfix_735572_adv_resnet-e78-acc0.7346.ckpt')
    #path = os.path.join(path, 'checkpoints', 'bnfix_225411_yuv_resnet-e99-acc0.7448.ckpt')
    #path = os.path.join(path, 'checkpoints', 'bnfix_551110_hsv_resnet-e70-acc0.6783.ckpt')

    with graph.as_default():
        saver.restore(sess, path)  # tf.train.latest_checkpoint(path))

    if foolbox:
        with sess.as_default():
            fmodel = TensorFlowModel(images, logits, bounds=(0, 255))
        return fmodel
    else:
        return images, logits, sess
