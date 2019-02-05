
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor

import foolbox
import numpy as np
import timeit

from models.utils.batch_tensorflow_model import BatchTensorflowModel
from models.utils.ensemble_tf_model import EnsembleTFModel
from utils.sampling.normal import sample_hypersphere


class BiasedBoundaryAttack:
    """
     Like BoundaryAttack, but uses biased sampling from prior beliefs (lucky guesses).

     Apart from Perlin Noise and projected gradients, this implementation contains more work that is not in the paper:
     - We try addidional patterns (single-pixel modification, jitter patterns) to escape local minima whenever the attack gets stuck
     - We dynamically tune hyperparameters according to the observed success of previous samples
     - At each step, multiple gradients are calculated to counter stochastic defenses
     - Optimized for speed: only use gradients if we can't progress without them.

    """

    def __init__(self, blackbox_model, sample_gen, substitute_model=None):
        """
        Creates a reusable instance.
        :param blackbox_model: The model to attack.
        :param sample_gen: Random sample generator.
        :param substitute_model: A surrogate model for gradients - either a TensorFlowModel, BatchTensorFlowModel or EnsembleTFModel.
        """

        self.blackbox_model = blackbox_model
        self.sample_gen = sample_gen

        self._jitter_mask = self.precalc_jitter_mask()

        # A substitute model that provides batched gradients.
        self.batch_sub_model = None
        if substitute_model is not None:
            if isinstance(substitute_model, foolbox.models.TensorFlowModel):
                self.batch_sub_model = BatchTensorflowModel(substitute_model._images, substitute_model._batch_logits, session=substitute_model.session)
            else:
                assert isinstance(substitute_model, EnsembleTFModel) or isinstance(substitute_model, BatchTensorflowModel)
                self.batch_sub_model = substitute_model

        # We use ThreadPools to calculate candidates and surrogate gradients while we're waiting for the model's next prediction.
        self.pg_thread_pool = ThreadPoolExecutor(max_workers=1)
        self.candidate_thread_pool = ThreadPoolExecutor(max_workers=1)

    def __enter__(self):
        self.pg_thread_pool.__enter__()
        self.candidate_thread_pool.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Will block until the futures are calculated. Thankfully they're not very complicated.
        self.pg_thread_pool.__exit__(exc_type, exc_value, traceback)
        self.candidate_thread_pool.__exit__(exc_type, exc_value, traceback)
        print("BiasedBoundaryAttack: all threads stopped.")

    def run_attack(self, X_orig, label, is_targeted, X_start, n_calls_left_fn, n_max_per_batch=50, n_seconds=None,
                   source_step=1e-2, spherical_step=1e-2, give_up_calls_left=0, give_up_dist=9999):
        """
        Runs the Biased Boundary Attack against a single image.
        The attack terminates when n_calls_left_fn() returns 0, n_seconds have elapsed, or a "give up" condition is reached.

        Give-up functionality:
        - When few calls are remaining, but the distance is still high. Could use the additional time for other images.
        - Could theoretically be used to game the final score: spend more time on imgs that will reduce the median, and give up on others
        - Largely unused (didn't get to finish this)

        :param X_orig: The original (clean) image to perturb.
        :param label: The target label (if targeted), or the original label (if untargeted).
        :param is_targeted: True if targeted.
        :param X_start: The starting point (must be of target class).
        :param n_calls_left_fn: A function that returns the currently remaining number of queries against the model.
        :param n_max_per_batch: How many samples are drawn per "batch". Samples are processed serially (the challenge doesn't allow
                                batching), but for each "batch", the attack dynamically adjusts hyperparams based on the success of
                                previous samples. This "batch" size is the max number of samples after which hyperparams are reset, and
                                a new "batch" is started. See generate_candidate().
        :param n_seconds: Maximum seconds allowed for the attack to complete.
        :param source_step: source step hyperparameter (see Boundary Attack)
        :param spherical_step: orthogonal step hyperparameter (see Boundary Attack)
        :param give_up_calls_left: give-up condition: if less than this number of calls is left
        :param give_up_dist: give-up condition: if the current L2 distance is higher than this
        :return: The best adversarial example so far.
        """

        assert len(X_orig.shape) == 3
        assert len(X_start.shape) == 3
        assert X_orig.dtype == np.float32

        time_start = timeit.default_timer()

        pg_future = None
        try:
            # WARN: Inside this function, image space is normed to [0,1]!
            X_orig = np.float32(X_orig) / 255.
            X_start = np.float32(X_start) / 255.

            label_current, dist_best = self._eval_sample(X_start, X_orig)
            if (label_current == label) != is_targeted:
                print("WARN: Starting point is not a valid adversarial example! Continuing for now.")

            X_adv_best = np.copy(X_start)

            # Abort if we're running out of queries
            while n_calls_left_fn() > 3:

                # Determine how many samples to draw at the current position.
                n_candidates = min(n_max_per_batch, n_calls_left_fn())

                # Calculate the projected adversarial gradient at the current position.
                #  Putting this into a ThreadPoolExecutor. While this is processing, we can already draw ~2 samples without waiting for the
                #  gradient. If the first 2 samples were unsuccessful, then the later ones can be biased with the gradient.
                # Also cancel any pending requests from previous steps.
                if pg_future is not None:
                    pg_future.cancel()
                pg_future = self.pg_thread_pool.submit(self.get_projected_gradients, **{
                    "x_current": X_adv_best,
                    "x_orig": X_orig,
                    "label": label,
                    "is_targeted": is_targeted})

                # Also do candidate generation with a ThreadPoolExecutor. We need to squeeze out every bit of runtime.
                # Queue the first candidate.
                candidate_future = self.candidate_thread_pool.submit(self.generate_candidate, **{
                    "i": 0,
                    "n": n_candidates,
                    "x_orig": X_orig,
                    "x_current": X_adv_best,
                    "source_step": source_step,
                    "spherical_step": spherical_step,
                    "pg_future": pg_future})

                for i in range(n_candidates):
                    # Get candidate and queue the next one.
                    candidate = candidate_future.result()
                    if i < n_candidates - 1:
                        candidate_future = self.candidate_thread_pool.submit(self.generate_candidate, **{
                            "i": i+1,
                            "n": n_candidates,
                            "x_orig": X_orig,
                            "x_current": X_adv_best,
                            "source_step": source_step,
                            "spherical_step": spherical_step,
                            "pg_future": pg_future})

                    time_elapsed = timeit.default_timer() - time_start
                    if n_seconds is not None and time_elapsed >= n_seconds:
                        print("WARN: Running out of time! Aborting attack!")
                        return X_adv_best * 255.

                    if dist_best > give_up_dist and n_calls_left_fn() < give_up_calls_left:
                        print("Distance is way too high, aborting attack to save time.")
                        return X_adv_best * 255.

                    # Test if successful. NOTE: dist is rounded here!
                    candidate_label, rounded_dist = self._eval_sample(candidate, X_orig)
                    unrounded_dist = np.linalg.norm(candidate - X_orig)
                    if (candidate_label == label) == is_targeted:
                        if unrounded_dist < dist_best:
                            print("@ {:.3f}: After {} samples, found something @ {:.3f} (rounded {:.3f})! (reduced by {:.1%})".format(
                                dist_best, i, unrounded_dist, rounded_dist, 1.-rounded_dist/dist_best))

                            # Terminate this batch (don't try the other candidates) and advance.
                            X_adv_best = candidate
                            dist_best = unrounded_dist
                            break

            return X_adv_best * 255.

        finally:
            # Be safe and wait for the gradient future. We want to be sure that no BG worker is blocking the GPU before returning.
            if pg_future is not None:
                futures.wait([pg_future])

    def generate_candidate(self, i, n, x_orig, x_current, source_step, spherical_step, pg_future):

        # This runs in a loop (while i<n) per "batch".
        # Whenever a candidate is successful, a new batch is started. Therefore, i is the number of previously unsuccessful samples.
        # Trying to use this in our favor, we tune our hyperparameters based on i:
        # - As i gets higher, progressively reduce step size for the next candidate
        # - When i gets high, try to blend jitter patterns and single pixels

        # Try this only once: blend a jitter pattern that brings us closer to the source,
        # but should be invisible to the defender (if they use denoising).
        if i == int(0.7 * n):
            candidate = x_current
            fade_eps = 0.005
            while np.sum(np.abs(np.round(candidate*255.) - np.round(x_current*255.))) < 0.0001:
                #print("jitter at i={} with fade_eps={}".format(i, fade_eps))
                candidate = self.generate_jitter_sample(x_orig, x_current, fade_eps=fade_eps)
                fade_eps += 0.005
            return candidate

        # Last resort: change single pixels to rip us out of the local minimum.
        i_pixel_start = int(0.9 * n)
        if i >= i_pixel_start:
            l0_pixel_index = i - i_pixel_start
            #print("pixel at {}".format(l0_pixel_index))
            candidate = self.generate_l0_sample(x_orig, x_current, n_px_to_change=1, px_index=l0_pixel_index)
            return candidate

        # Default: use the BBA. Scale both spherical and source step with i.
        scale = (1. - i/n) + 0.3
        c_source_step = source_step * scale
        c_spherical_step = spherical_step * scale

        # Get the adversarial projected gradient from the (other) BG worker.
        #  Create the first 2 candidates without it, so we can already start querying the model. The BG worker can finish the gradients
        #  while we're waiting for those first 2 results.
        pg_factor = 0.5
        pgs = None
        if i >= 2:
            # if pg_future.running():
            #     print("Waiting for gradients...")
            pgs = pg_future.result()
        pgs = pgs if i % 2 == 0 else None           # Only use gradient bias on every 2nd iteration.

        candidate, spherical_candidate = self.generate_boundary_sample(
            X_orig=x_orig, X_adv_current=x_current, source_step=c_source_step, spherical_step=c_spherical_step,
            sampling_fn=self.sample_gen.get_perlin, pgs_current=pgs, pg_factor=pg_factor)

        return candidate

    def generate_l0_sample(self, X_orig, X_aex, n_px_to_change=1, px_index=0):
        # Modified copypasta from refinement_tricks.refine_jitter().
        # Change the n-th important pixel.

        # Sort indices of the pixels, descending by difference to original.
        # TODO: try color-triples?
        i_highest_diffs = np.argsort(np.abs(X_aex - X_orig), axis=None)[::-1]

        X_candidate = X_aex.copy()

        # Try and replace n pixels at once.
        i_pxs = i_highest_diffs[px_index: px_index + n_px_to_change]
        for i_px in i_pxs:
            i_px = np.unravel_index(i_px, X_orig.shape)
            X_candidate[i_px] = X_orig[i_px]

        return X_candidate

    def precalc_jitter_mask(self):
        # Prepare a jitter mask with XOR (alternating). TODO: we could really improve this pattern. S&P noise, anyone?
        jitter_width = 5
        jitter_mask = np.empty((64, 64, 3), dtype=np.bool)
        for i in range(64):
            for j in range(64):
                jitter_mask[i, j, :] = (i % jitter_width == 0) ^ (j % jitter_width == 0)
        return jitter_mask

    def generate_jitter_sample(self, X_orig, X_aex, fade_eps=0.01):
        # Modified copypasta from refinement_tricks.refine_pixels().

        jitter_mask = self._jitter_mask

        jitter_diff = np.zeros(X_orig.shape, dtype=np.float32)
        jitter_diff[jitter_mask] = (X_aex - X_orig)[jitter_mask]

        X_candidate = X_aex - fade_eps * jitter_diff
        return X_candidate

    def generate_boundary_sample(self, X_orig, X_adv_current, source_step, spherical_step, sampling_fn, pgs_current=None, pg_factor=0.3):
        # Partially adapted from FoolBox BoundaryAttack.

        unnormalized_source_direction = np.float64(X_orig) - np.float64(X_adv_current)
        source_norm = np.linalg.norm(unnormalized_source_direction)
        source_direction = unnormalized_source_direction / source_norm

        # Get perturbation from provided distribution
        sampling_dir = sampling_fn()

        # ===========================================================
        # calculate candidate on sphere
        # ===========================================================
        dot = np.vdot(sampling_dir, source_direction)
        sampling_dir -= dot * source_direction                                          # Project orthogonal to source direction
        sampling_dir /= np.linalg.norm(sampling_dir)

        # If available: Bias the spherical dirs in direction of the adversarial gradient, which is projected onto the sphere
        if pgs_current is not None:

            # We have a bunch of gradients that we can try. Randomly select one.
            # NOTE: we found this to perform better than simply averaging the gradients.
            pg_current = pgs_current[np.random.randint(0, len(pgs_current))]

            sampling_dir = (1. - pg_factor) * sampling_dir + pg_factor * pg_current
            sampling_dir /= np.linalg.norm(sampling_dir)
            sampling_dir *= spherical_step * source_norm                                # Norm to length stepsize*(dist from src)

        D = 1 / np.sqrt(spherical_step ** 2 + 1)
        direction = sampling_dir - unnormalized_source_direction
        spherical_candidate = X_orig + D * direction

        np.clip(spherical_candidate, 0., 1., out=spherical_candidate)

        # ===========================================================
        # step towards source
        # ===========================================================
        new_source_direction = X_orig - spherical_candidate
        new_source_direction_norm = np.linalg.norm(new_source_direction)

        # length if spherical_candidate would be exactly on the sphere
        length = source_step * source_norm
        # length including correction for deviation from sphere
        deviation = new_source_direction_norm - source_norm
        length += deviation

        # make sure the step size is positive
        length = max(0, length)

        # normalize the length
        length = length / new_source_direction_norm

        candidate = spherical_candidate + length * new_source_direction
        np.clip(candidate, 0., 1., out=candidate)

        return np.float32(candidate), np.float32(spherical_candidate)

    def get_projected_gradients(self, x_current, x_orig, label, is_targeted):
        # Idea is: we have a direction (spherical candidate) in which we want to sample.
        # We know that the gradient of a substitute model, projected onto the sphere, usually points to an adversarial region.
        # Even if we are already adversarial, it should point "deeper" into that region.
        # If we sample in that direction, we should move toward the center of the adversarial cone.
        # Here, we simply project the gradient onto the same hyperplane as the spherical samples.
        #
        # Instead of a single projected gradient, this method returns an entire batch of them:
        # - Surrogate gradients are unreliable, so we sample them in a region around the current position.
        # - This gives us a similar benefit as observed "PGD with random restarts".

        source_direction = x_orig - x_current
        source_norm = np.linalg.norm(source_direction)
        source_direction = source_direction / source_norm

        # Take a tiny step towards the source before calculating the gradient. This marginally improves our results.
        step_inside = 0.002 * source_norm
        x_inside = x_current + step_inside * source_direction

        # Perturb the current position before calc'ing gradients
        n_samples = 8
        radius_max = 0.01 * source_norm
        x_perturb = sample_hypersphere(n_samples=n_samples, sample_shape=x_orig.shape, radius=1, sample_gen=self.sample_gen)
        x_perturb *= np.random.uniform(0., radius_max)

        x_inside_batch = x_inside + x_perturb

        gradients = (self.batch_sub_model.gradient(x_inside_batch * 255., [label] * n_samples) / 255.)
        if is_targeted:
            gradients = -gradients

        # Project the gradients.
        for i in range(n_samples):
            dot = np.vdot(gradients[i], source_direction)
            projected_gradient = gradients[i] - dot * source_direction              # Project orthogonal to source direction
            projected_gradient /= np.linalg.norm(projected_gradient)                # Norm to length 1
            gradients[i] = projected_gradient
        return gradients

    def _eval_sample(self, x, x_orig_normed=None):
        # Round, then get label and distance.
        x_rounded = np.round(np.clip(x * 255., 0, 255))
        preds = self.blackbox_model.predictions(np.uint8(x_rounded))
        label = np.argmax(preds)

        if x_orig_normed is None:
            return label
        else:
            dist = np.linalg.norm(x_rounded/255. - x_orig_normed)
            return label, dist
