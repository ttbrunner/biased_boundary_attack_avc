"""
This is the ResNet18 baseline from the competition organizers:
https://gitlab.crowdai.org/adversarial-vision-challenge/resnet18_model_baseline
"""
import tensorflow as tf
import os

from foolbox.models import TensorFlowModel
from models.resnet18_baseline.resnet_model import Model


def _create_resnet(graph=None, x_input=None):

    if graph is None:
        graph = tf.Graph()

    with graph.as_default():
        images = x_input if x_input is not None else tf.placeholder(tf.float32, (None, 64, 64, 3))

        # preprocessing
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        _CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
        features = images - tf.constant(_CHANNEL_MEANS)

        # Stupid vars counting trick to allow ensembles in the root scope. TF variable scopes suck!
        vars_before = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)
        model = Model(
            resnet_size=18,
            bottleneck=False,
            num_classes=200,
            num_filters=64,
            kernel_size=3,
            conv_stride=1,
            first_pool_size=0,
            first_pool_stride=2,
            second_pool_size=7,
            second_pool_stride=1,
            block_sizes=[2, 2, 2, 2],
            block_strides=[1, 2, 2, 2],
            final_size=512,
            version=2,
            data_format=None)

        logits = model(features, False)

        vars_after = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)
        vars_new = set(vars_after).difference(vars_before)

        with tf.variable_scope('utilities'):
            saver = tf.train.Saver(list(vars_new))

    return graph, saver, images, logits


def create_rn18_model(sess=None, x_input=None, foolbox=True):

    # Allow to reuse a session and put the model on top of an existing input
    assert (sess is not None) == (x_input is not None)

    if sess is not None:
        graph, saver, images, logits = _create_resnet(sess.graph, x_input)
    else:
        graph, saver, images, logits = _create_resnet()
        sess = tf.Session(graph=graph)

    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, 'checkpoints', 'model')
    saver.restore(sess, tf.train.latest_checkpoint(path))

    if foolbox:
        with sess.as_default():
            fmodel = TensorFlowModel(images, logits, bounds=(0, 255))
        return fmodel
    else:
        return images, logits, sess
