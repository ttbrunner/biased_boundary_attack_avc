import tensorflow as tf


class BatchTensorflowModel:
    # Tiny helper (NOT a foolbox model!) that allows us to get gradients in a batch. Foolbox doesn't allow that.

    def __init__(self, images, logits, session=None):
        # We want to reuse the session in which the other stuff was put into.
        # If this is the wrong session then we crash without explanation.

        self._created_session = False
        if session is None:
            session = tf.get_default_session()
            if session is None:
                session = tf.Session(graph=images.graph)
                self._created_session = True

        with session.graph.as_default():
            self._session = session
            self._images = images
            self._logits = logits
            self._labels = tf.placeholder(tf.int64, shape=None, name='labels')

            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self._labels,
                logits=self._logits)

            gradients = tf.gradients(self.loss, self._images)
            self._gradient = gradients[0]

    def __exit__(self, exc_type, exc_value, traceback):
        if self._created_session:
            self._session.close()
        return None

    @property
    def session(self):
        return self._session

    def batch_predictions(self, images):
        predictions = self._session.run(
            self._logits,
            feed_dict={self._images: images})
        return predictions

    def gradient(self, images, labels):
        g = self._session.run(
            self._gradient,
            feed_dict={
                self._images: images,
                self._labels: labels})
        return g
