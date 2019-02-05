import numpy as np
from foolbox.models import Model
from adversarial_vision_challenge.common import check_image as avc_check_image

from utils.util import hash_image


class RemoteModelWrapper(Model):
    """
    This wrapper makes sure that we save exactly the same images that we checked against the model.
    It's easy to introduce bugs into the attack by making rounding errors, and then finally saving an image that is
    actually not adversarial. Also, it's useful for local testing because it emulates the conditions of the challenge 1:1.

    - Rounds&clips every image before sending it to the model, and removes logits from the result
    - Keeps track of the best adversarial and its result
    - Can return this best result, so no matter what the attack code does, we always have a reliable image that worked.
    - Caches images->pred mappings in case an attack submits the same image multiple times.

    Note on ModelWrapper:
        I had some weird problem when trying to inherit from ModelWrapper, in that somehow the virtual
        predictions() would call ModelWrapper.batch_predictions(), instead of (this).batch_predictions().
        To be safe, we do all of the inheritance by ourselves.
    """

    def __init__(self, model, do_hash=True):
        super(RemoteModelWrapper, self).__init__(
            bounds=model.bounds(),
            channel_axis=model.channel_axis())

        self.wrapped_model = model

        self._adv_orig_img = None
        self._adv_is_targeted = False
        self._adv_label = None
        self._adv_best_img = None
        self._adv_best_dist = None
        self._adv_n_calls = 0               # Count number of calls since the last adversarial target was set.

        # Image hash -> class id. Since the model MUST be deterministic, we may as well cache the intput-output mapping.
        # Foolbox likes to query the clean image (or adversarial starting point) multiple times, so we can save time and tries like this.
        self._do_hash = do_hash
        self._pred_hashtable = {}

    def __enter__(self):
        assert self.wrapped_model.__enter__() == self.wrapped_model
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.wrapped_model.__exit__(exc_type, exc_value, traceback)

    def num_classes(self):
        return self.wrapped_model.num_classes()

    def batch_predictions(self, images):
        """
        Runs a batch prediction on the model. NOTE: len(images) must be 1! It's not a batch, but the online evaluation always works on
        only 1 image. Therefore, we force the caller to emulate the same behaviour.

        For convenience, use predictions() instead.
        :param images: must have size (1, 64, 64, 3)
        :return: One-hot prediction vector without logits (i.e. only 1 and 0's).
        """
        assert images.shape[0] == 1, 'I know this is called "batch", but we are exactly emulating the conditions of the online evaluation.'

        # Repeat the checks from AVC here, to make sure we don't save a "wrong" image.
        img_to_eval = avc_check_image(images[0])

        if self._do_hash:
            # If the image is in the hashtable, return it as it is.
            # We hash every image, but this implementation is very cheap.
            im_hash = hash_image(img_to_eval)
            clsid = self._pred_hashtable.get(im_hash, None)
            if clsid is not None:
                preds_cleaned = np.zeros(self.num_classes(), dtype=np.float32)
                preds_cleaned[clsid] = 1.
                return preds_cleaned[np.newaxis, :]

        preds = self.wrapped_model.predictions(img_to_eval)
        clsid = np.argmax(preds)

        if self._do_hash:
            # Remember all hashes and predictions.
            # Should be less than 30MB for a single evaluation run.
            self._pred_hashtable[im_hash] = clsid

        # Clean up the preds (1 0 0 0 etc), to make sure we can't use logits even in local testing.
        preds_cleaned = np.zeros(self.num_classes(), dtype=np.float32)
        preds_cleaned[clsid] = 1.

        # Remember the best img
        if self._adv_orig_img is not None:
            self._adv_n_calls += 1

            correct = (clsid == self._adv_label)
            if self._adv_is_targeted == correct:                # XAND - if targeted and correct, or not targeted and not correct.
                dist = np.linalg.norm(np.float32(self._adv_orig_img) / 255. - np.float32(img_to_eval) / 255.)
                if dist < self._adv_best_dist:
                    self._adv_best_dist = dist
                    self._adv_best_img = img_to_eval

        return preds_cleaned[np.newaxis, :]

    def adv_set_target(self, orig_img, is_targeted, label):
        """
        Sets a new adversarial target and resets the n_calls counter.
        :param orig_img: The original image being perturbed
        :param is_targeted: If true, then the attack is successful if pred==label.
                            If false, then the attack is succesful if pred!=label.
        :param label: If is_targeted is True, then the target adversarial class.
                      If is_targeted is False, then the original (correct) class.
        """
        self._adv_orig_img = orig_img
        self._adv_is_targeted = is_targeted
        self._adv_label = label
        self._adv_best_img = None
        self._adv_best_dist = 99999.
        self._adv_n_calls = 0
        self._pred_hashtable.clear()                # Clear for each image - collisions could be a real problem otherwise

    def adv_get_best_img(self):
        return self._adv_best_img

    def adv_get_n_calls(self):
        return self._adv_n_calls
