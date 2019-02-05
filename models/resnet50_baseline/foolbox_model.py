"""
This is the ResNet50-ALP baseline from the competition organizers:
https://gitlab.crowdai.org/adversarial-vision-challenge/resnet50_alp_model_baseline
"""

import tensorflow as tf
import os
from foolbox.models import TensorFlowModel

from . import my_resnet


def model_fn(images, is_training, reuse=tf.AUTO_REUSE):
    with tf.contrib.framework.arg_scope(my_resnet.resnet_arg_scope()):
        resnet_fn = my_resnet.resnet_v2_50
        logits, _ = resnet_fn(images, 200, is_training=is_training,
                              reuse=reuse)
        logits = tf.reshape(logits, [-1, 200])
    return logits


def _create_model(graph=None, x_input=None):
    if graph is None:
        graph = tf.Graph()

    with graph.as_default():

        images = x_input if x_input is not None else tf.placeholder(tf.float32, (None, 64, 64, 3))
        features = (images / 255. - 0.5) * 2.

        # Need to do this crazy variable counting trick, because the checkpoint was saved without a variable scope.
        #  Therefore, the saver always tries to restore all variables in the root scope.
        #  If we make an ensemble (in the same graph), this would interfere with the other models (that are also not scoped...)
        vars_before = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)
        logits = model_fn(features, is_training=False)
        vars_after = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)
        vars_new = set(vars_after).difference(vars_before)

        with tf.variable_scope('utilities'):
            saver = tf.train.Saver(list(vars_new))

    return graph, saver, images, logits


def create_rn50_model(sess=None, x_input=None, foolbox=True):

    # Allow to reuse a session and put the model on top of an existing input
    assert (sess is not None) == (x_input is not None)

    if sess is not None:
        graph, saver, images, logits = _create_model(sess.graph, x_input)
    else:
        graph, saver, images, logits = _create_model(None, None)
        sess = tf.Session(graph=graph)

    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, 'tiny_imagenet_alp05_2018_06_26.ckpt')
    saver.restore(sess, path)  # tf.train.latest_checkpoint(path))

    if foolbox:
        with sess.as_default():
            fmodel = TensorFlowModel(images, logits, bounds=(0, 255))
        return fmodel
    else:
        return images, logits, sess
