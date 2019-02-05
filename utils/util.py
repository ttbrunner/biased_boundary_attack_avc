import numpy as np
import hashlib
import scipy

from foolbox.distances import Distance


def eval_distance(X, Y):
    """
    Calculates the L2 distance between two images.
    Copypasta from https://github.com/bethgelab/adversarial-vision-challenge/blob/master/bin/avc-test-attack,
     so we can be sure it is the same.
    """
    # assert X.dtype == np.uint8             # Allow float32, although the real (rounded&cropped) distance could be different.
    # assert Y.dtype == np.uint8
    assert X.shape == Y.shape

    X = X.astype(np.float64) / 255
    Y = Y.astype(np.float64) / 255
    return np.linalg.norm(X - Y)


def eval_distance_rounded(X, Y):
    return eval_distance(np.clip(np.round(X), 0, 255), np.clip(np.round(Y), 0, 255))


def hash_image(x):
    """
    Hashes an image into a 32-bit integer.
    """
    assert x.dtype == np.uint8
    b = x.tobytes()
    h = hashlib.sha256(b)
    hd = h.digest()
    s = int.from_bytes(hd, byteorder='big') % 2**32
    return s


def get_seed_from_img(x):
    """
    Hashes an image and returns an int32 for use with the RNG
    """
    assert x.shape == (64, 64, 3)
    b = x.tobytes()
    h = hashlib.sha256(b)
    hd = h.digest()
    s = int.from_bytes(hd, byteorder='big')
    s *= 9001           # Try to make sure somebody else doesn't guess our seed. This is really paranoid, I know.
    s = s % 2**32
    return s


def softmax(logits):
    """
    Compute softmax values for each sets of scores in x. Can be one- or two-dimensional.
    """
    if len(logits.shape) == 1:
        logits = logits - np.max(logits)
        e = np.exp(logits)
        return e / np.sum(e)

    elif len(logits.shape) == 2:
        logits = logits - np.max(logits, axis=1, keepdims=True)
        e = np.exp(logits)
        return e / np.sum(e, axis=1, keepdims=True)

    else:
        raise ValueError("Need to provide either an 1D or 2D array.")


def vec_cdf(x, m, s):
    """
    Vectorized CDF function, that can apply a different distribution to each x.
    x, m and s should have the same dimensionality.
    """
    z = (x - m) / s
    return scipy.stats.norm.cdf(z)


class FoolboxL2Distance(Distance):
    """
    Foolbox implementation of the "official" L2 evaluation metric used in the competition. This is not MSE, which divides by n.
    Can be used with any Foolbox attack. It's really just for debugging, so the output is more readable.
    """

    def _calculate(self):
        min_, max_ = self._bounds
        f = (max_ - min_)**2

        diff = self.other - self.reference
        value = np.sqrt(np.vdot(diff, diff) / f)

        # calculate the gradient only when needed
        self._g_diff = diff
        self._g_f = f
        gradient = None
        return value, gradient

    @property
    def gradient(self):
        if self._gradient is None:
            # Pretty sure this gradient isn't really correct..... ...  TODO: if ever we use gradient-based attacks with this, double check!
            self._gradient = np.sqrt(self._g_diff / (self._g_f / 2))
        return self._gradient

    def __str__(self):
        return 'normalized MSE = {:.2e}'.format(self._value)
