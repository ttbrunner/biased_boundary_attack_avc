import tensorflow as tf

from models.denoising.dunet_denoiser import DunetDenoiser
from models.resnet18_baseline.foolbox_model import create_rn18_model
from models.resnet50_baseline.foolbox_model import create_rn50_model
from models.inception_resnet_smaller.foolbox_model import create_ir_model


class EnsembleTFModel:
    """
    Combines multiple models (same input, fused output). Provides gradients for the ensemble.
    Not a FoolBox model, but similar interface.
    """

    def __init__(self, session=None, x_input=None, with_denoiser=False):

        self._created_session = False
        if session is None:
            if x_input is None:
                graph = tf.Graph()
                session = tf.Session(graph=graph)
            else:
                session = tf.Session()
            self._created_session = True
        self._session = session

        self._n_classes = 200

        if with_denoiser:
            # Add a denoiser to the model graph. Then we can get gradients for the entire ensemble + denoiser.
            self._denoiser = DunetDenoiser(session, x_input)
            self._images = self._denoiser._tf_x
        else:
            self._denoiser = None
            self._images = x_input

        self._create_ensemble()

    def __exit__(self, exc_type, exc_value, traceback):
        if self._created_session:
            self._session.close()
        return None

    def _create_ensemble(self):

        with self._session.graph.as_default():

            if self._images is None:
                self._images = tf.placeholder(tf.float32, (None, 64, 64, 3))

            if self._denoiser is not None:
                ens_input = self._denoiser._tf_x - self._denoiser._tf_noise
            else:
                ens_input = self._images

            _, fb_rn50_logits, _ = create_rn50_model(sess=self._session, x_input=ens_input, foolbox=False)
            _, fb_rn18_logits, _ = create_rn18_model(sess=self._session, x_input=ens_input, foolbox=False)
            _, fb_irn_logits, _ = create_ir_model(sess=self._session, x_input=ens_input, foolbox=False)

            self.net_logits = [fb_rn50_logits, fb_irn_logits, fb_rn18_logits]

            # The entire model works on logits only. But we also include fused softmax for those that want them.
            # Fuse logits (use special scaling). These will be needed for gradients (transfer attack)
            # Ensemble weighting: rn50 ALP has very compact logits, so we need to sqrt the other participants to keep them from dominating.
            self.fused_logits = (fb_rn50_logits +
                                 tf.sqrt(tf.abs(fb_irn_logits)) * tf.sign(fb_irn_logits) +
                                 tf.sqrt(tf.abs(fb_rn18_logits)) * tf.sign(fb_rn18_logits)) / 4.

            # Fuse softmax (use normal weighted scaling). Unused for now.
            self.fused_softmax = (tf.nn.softmax(fb_rn50_logits) + tf.nn.softmax(fb_irn_logits) + tf.nn.softmax(fb_rn18_logits)) / 3.

            self._labels = tf.placeholder(tf.int64, shape=None, name='labels')
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self._labels,
                logits=self.fused_logits)

            gradients = tf.gradients(self.loss, self._images)
            self._gradient = gradients[0]

    @property
    def session(self):
        return self._session

    def num_classes(self):
        return self._n_classes

    def batch_predictions(self, images):
        predictions = self._session.run(self.fused_logits,
            feed_dict={self._images: images})

        return predictions

        # predictions = self._session.run(
        #     [self.fused_logits,] + self.net_logits,
        #     feed_dict={self._images: images})
        #
        # # Print logit magnitude for debug
        # print("Mag-fused: {}".format(np.max(predictions[0])))
        # print("Mag-50: {}".format(np.max(predictions[1])))
        # print("Mag-irn: {}".format(np.max(predictions[2])))
        # print("Mag-18: {}".format(np.max(predictions[3])))
        # return predictions[0]

    def gradient(self, images, labels):
        g = self._session.run(
            self._gradient,
            feed_dict={
                self._images: images,
                self._labels: labels})
        return g
