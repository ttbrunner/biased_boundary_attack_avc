import numpy as np
import foolbox
import timeit
import itertools

from attacks.methods import mod_boundary_attack
from attacks.methods.biased_boundary_attack import BiasedBoundaryAttack
from attacks.methods.super_fgm import SuperFGM
from models.utils.ensemble_tf_model import EnsembleTFModel
from utils import util


class MultiBoundaryAttack:
    """
    Targeted and untargeted attack, based on BiasedBoundaryAttack.
    - Picks multiple promising starting points from known images of the target class
    - Ranks them according to estimated success probability (see get_target_class_imgs())
    - Does a quick BoundaryAttack on all starting points
    - Chooses the best candidate and runs our own BiasedBoundaryAttack until all calls are spent.

    Alternatively, can also use a PGD-style attack to find a starting point. Currently, I'm not entirely sure if this improves or degrades
    the success chances (see discussion below).
    """

    def __init__(self, model, X_known, y_known, n_classes, sample_gen=None, cached_conf=None, with_denoiser=False):
        """
        Creates a reusable instance of the MultiBoundaryAttack.
        :param model: the model to attack.
        :param X_known: a dataset of images with known labels, from which starting points are picked.
        :param y_known: a dataset of images with known labels, from which starting points are picked.
        :param n_classes: number of classes.
        :param sample_gen: reusable random sample generator
        :param cached_conf: optional, precalculated class confidence values for the known dataset. See get_target_class_imgs()
        :param with_denoiser: if True, adds a denoiser to the surrogate model for calculating gradients. Doesn't seem to be effective.
        """

        self.remote_model = model
        self.X_known = X_known
        self.y_known = y_known
        self.n_classes = n_classes
        self.sample_gen = sample_gen
        self.cached_conf = cached_conf

        assert hasattr(model, "adv_get_n_calls"), "Please wrap the model in a RemoteModelWrapper."

        # Preprocess img ids into buckets by class. list(n_classes)[list(img_ids per class)]
        self.clsid_imgid_buckets = [[] for _ in range(n_classes)]
        for i in range(len(X_known)):
            self.clsid_imgid_buckets[y_known[i]].append(i)

        # Create a substitute model.
        # Use either BatchTFModel or EnsembleTFModel, because the default Foolbox TFModels don't provide batched gradients.
        self.substitute_model = EnsembleTFModel(with_denoiser=with_denoiser)

        # Create the BBA attack. It has ThreadPools, so we need to use the context manager for cleanup.
        self.bba_attack = BiasedBoundaryAttack(self.remote_model, sample_gen=self.sample_gen, substitute_model=self.substitute_model)
        self._has_valid_context = False

        self._stats_n_total = 0
        self._stats_n_successful_fgm = 0
        self._stats_override_disable_fgm = False

    def __enter__(self):
        self.bba_attack.__enter__()
        self._has_valid_context = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._has_valid_context = False
        return self.bba_attack.__exit__(exc_type, exc_value, traceback)

    def sort_by_distance(self, arr, original_img, dist_penalties=None):
        # Sorts an array of image ids by distance to the original image.
        # If dist_penalties is provided, then for each image the distance is weighted by an individual penalty.
        # NOTE: this loop is not optimized, but for now it's just bearable. For larger datasets we should precalc L2 norms.

        timer = timeit.default_timer()
        orig_img_norm = np.float32(original_img) / 255.

        n = len(arr)
        dists = np.empty(n, dtype=np.float32)
        for i in range(n):
            dists[i] = np.linalg.norm(np.float32(arr[i]) / 255. - orig_img_norm)

        if dist_penalties is not None:
            assert len(dist_penalties) == len(arr)
            dists *= dist_penalties

        sorting = np.argsort(dists)

        elapsed_s = timeit.default_timer() - timer
        print("Sorted {} elements in {} seconds.".format(len(arr), elapsed_s))
        return arr[sorting], dists[sorting]

    def get_target_class_imgs(self, label, is_targeted, model, original_img, n_imgs=1):
        # Get the n most promising starting points, based on distance to the source and classification confidence.

        # Get all images from the desired class
        if is_targeted:
            img_ids = np.array(self.clsid_imgid_buckets[label])
        else:
            img_ids = np.array(list(itertools.chain.from_iterable(self.clsid_imgid_buckets[clsid] for clsid in range(200) if clsid != label)))

        # Calc a penalty based on classification confidence. In our case, cached_conf simply contains precalculated softmax activations
        #  of ResNet50 for the entire dataset (X_known). The rationale is that if confidence is high, then the salient features
        #  of the target class must be very pronounced, and we are very far from the decision boundary. Therefore it is likely that we
        #  can modify the image alot, while staying in the adversarial region.
        if self.cached_conf is not None:
            cached_softmax = self.cached_conf[img_ids]
            if is_targeted:
                # Prefer images that were strongly classified as the target label.
                # Penalty = 1 - highest softmax activation (of the target label, naturally)
                dist_penalties = np.max(cached_softmax, axis=1)
                dist_penalties = np.sqrt(dist_penalties)            # Sqrt it so the heuristic isn't as strong
                dist_penalties /= np.max(dist_penalties)            # Norm
                dist_penalties = 1. - dist_penalties
            else:
                # Untargeted: penalize images that have (medium-)high softmax for the original label. Penalize images that have high
                # activation for the source label.
                # Penalty = 1 + softmax activation of the original label (starting at 1e-3 for precision)
                label_confs = cached_softmax[:, label]
                bad_candidate_filter = label_confs > 5e-4
                dist_penalties = np.ones(len(img_ids), np.float32)
                dist_penalties[bad_candidate_filter] = label_confs[bad_candidate_filter] * 2e3
        else:
            dist_penalties = None

        X_sorted, _ = self.sort_by_distance(self.X_known[img_ids], np.float32(original_img), dist_penalties=dist_penalties)

        imgs = []
        for i, image in enumerate(X_sorted):

            if (np.argmax(model.predictions(image)) == label) == is_targeted:
                print('found an image from the target class')
                imgs.append(np.float32(image))
                if len(imgs) >= n_imgs:
                    return np.array(imgs), i + 1, True
            else:
                print('ignoring image that was wrongly classified by model')
                continue

        if len(imgs) > 0:
            print('found only {} images from target class!'.format(len(imgs)))
            return np.array(imgs), len(img_ids), True

        print('could not find an image from the target class')
        # but we still return the last image, the interpolation could in theory still work
        return np.float32(np.array([self.X_known[0]])), len(img_ids), False

    def run_attack(self, image, label, is_targeted, start_with_fgm=False, fgm_acceptable_dist=7, time_max=89):
        """
        Runs the MultiBoundaryAttack with a source image until either the query limit or the time limit is reached.
        :param image: the source image (clean) for which an adversarial example is generated.
        :param label: the target label (if targeted), or the original label (if untargeted).
        :param is_targeted: True if targeted.
        :param start_with_fgm: if True, try a PGD transfer attack to determine a starting point.
        :param fgm_acceptable_dist: threshold for "lucky PGD result". If the PGD attack yielded an adversarial example below this
                                    distance, skip the other starting points and proceed directly with the final BBA attack.
        :param time_max: time limit in seconds.
        :return: an adversarial example, or None if no starting point could be found.
        """

        assert image.dtype == np.float32
        assert image.min() >= 0
        assert image.max() <= 255
        assert self._has_valid_context, "Need to use this with a ContextManager!"

        time_attack_start = timeit.default_timer()

        # Return if already classified according to our wishes. This happens often in the untargeted scenario.
        original_label = np.argmax(self.remote_model.predictions(image))
        if (original_label == label) == is_targeted:
            return image

        fb_criterion = foolbox.criteria.TargetClass(label) if is_targeted else foolbox.criteria.Misclassification()
        aex_best = None
        aex_best_dist = 9999

        # Try a PGD attack before everything else.
        # Sometimes, PGD will score a lucky hit that is better than what the BBA can find with clean starting points.
        # But often, we just waste our time. We also observed that starting points provided by PGD are of low quality,
        # and subsequent BBA gets stuck immediately. Therefore, we spend very few iterations on PGD, and only accept it
        # as a starting point if it yielded an extremely good result.
        # Additionally, if we see this failing too often, we disable it to have more queries for the BBA.
        if self._stats_n_total > 0:
            if start_with_fgm:
                print("FGM success rate so far: {:.1%}".format(self._stats_n_successful_fgm / self._stats_n_total))
        if self._stats_n_total > 15:
            if start_with_fgm:
                if not self._stats_override_disable_fgm and self._stats_n_successful_fgm / self._stats_n_total < 0.15:
                    print("Stats: Deactivated FGM because of low success rate.")
                    self._stats_override_disable_fgm = True
        self._stats_n_total += 1
        start_with_fgm = start_with_fgm and not self._stats_override_disable_fgm

        if start_with_fgm:
            dist_early_abort = 3. if is_targeted else 1.            # Extremely good result, immediately proceed to BBA
            stepsize_give_up = 15. if is_targeted else 11.          # Result will be worse than clean starting point, give up early

            super_fgm = SuperFGM(self.remote_model, substitute_model=self.substitute_model, original_image=image, label=label,
                                 is_targeted=is_targeted, sample_gen=self.sample_gen)
            aex_best, aex_best_dist = super_fgm.run_attack(do_refine=False, eps=1.3, stepsize=.75, eps_max=60., n_it_per_eps=12,
                                                           bs_iterations=14, random_start=True, random_start_perlin=True,
                                                           noise_on_it_scale=12, n_grad_samples=8, momentum=0.0,
                                                           dist_early_abort=dist_early_abort, stepsize_give_up=stepsize_give_up)
            if aex_best is not None:
                aex_best = np.round(np.clip(aex_best, 0., 255.))
            if aex_best_dist < fgm_acceptable_dist:
                self._stats_n_successful_fgm += 1

        # If PGD found nothing good, get 6 starting point candidates.
        if aex_best_dist > fgm_acceptable_dist:

            n_starting_points = 6
            starting_points, calls, is_adv = self.get_target_class_imgs(label=label, is_targeted=is_targeted, model=self.remote_model,
                                                                        original_img=image, n_imgs=n_starting_points)
            n_starting_points = len(starting_points)                                # In case we found less
            if not is_adv:
                print('Could not find a starting point, help :(')                   # Surely this would never happen... right?
                return None

            # HACK: If the PGD result was so-so (above threshold, but below 15), use it as one of the starting point candidates.
            if aex_best_dist < 15:
                if is_targeted:
                    # Count as successful if targeted. For untargeted, <15 is not considered successful.
                    self._stats_n_successful_fgm += 1
                starting_points[-1] = aex_best

            # For each starting point, start with a simple Boundary Attack - just to see how far we get in a few queries -
            # and then decide which one to follow.
            aex_best = starting_points[0]
            for i in range(n_starting_points):
                print("Starting Point {}:".format(i))

                attack = mod_boundary_attack.BoundaryAttack(self.remote_model, fb_criterion)
                adv_obj = foolbox.adversarial.Adversarial(self.remote_model, fb_criterion, original_image=image,
                                                          original_class=original_label, distance=util.FoolboxL2Distance)

                # Boundary Attack with 20% and then 10% source step, but small spherical step. Early stopping. This covers much of the
                #  early "cheap gains" of the Boundary Attack. After this, we have a good intuition of which candidate is promising.
                #  We hope to spend no more than ~20 queries per starting point.
                aex = attack(adv_obj, iterations=6, spherical_step=5e-2, source_step=2e-1, step_adaptation=1.5, max_directions=5,
                             tune_batch_size=False, starting_point=starting_points[i], stop_early_diff=0.05, sample_gen=self.sample_gen,
                             normal_factor=0.0)
                aex = attack(adv_obj, iterations=6, spherical_step=5e-2, source_step=1e-1, step_adaptation=1.5, max_directions=5,
                             tune_batch_size=False, stop_early_diff=0.05, sample_gen=self.sample_gen, normal_factor=0.0)

                dist = util.eval_distance(image, aex)
                if dist < aex_best_dist:
                    aex_best_dist = dist
                    aex_best = aex
        else:
            # We did not choose starting points because PGD was so good already.
            # Nevertheless, do another coarse BA iteration before launching into the BIM.
            print("Refining FGM result...")
            attack = mod_boundary_attack.BoundaryAttack(self.remote_model, fb_criterion)
            adv_obj = foolbox.adversarial.Adversarial(self.remote_model, fb_criterion, original_image=image, original_class=original_label,
                                                      distance=util.FoolboxL2Distance)
            aex_best = attack(adv_obj, iterations=6, spherical_step=5e-2, source_step=1e-1, step_adaptation=1.5, max_directions=5,
                              tune_batch_size=False, starting_point=aex_best, stop_early_diff=0.05, sample_gen=self.sample_gen,
                              normal_factor=0.0)

        # At this point, we shouldn't have used much more than 100-200 calls. Hopefully.
        n_calls_used_before_bia = self.remote_model.adv_get_n_calls()
        print("Used {} calls so far.".format(n_calls_used_before_bia))
        n_calls_max = 995

        print("Continuing with best AEx so far...")
        time_elapsed = timeit.default_timer() - time_attack_start
        time_left = time_max - time_elapsed
        give_up_dist = 999  # 15 if is_targeted else 7    # We deactivated "give up" for our final submission.
        aex = self.bba_attack.run_attack(X_orig=image, label=label, is_targeted=is_targeted, X_start=aex_best,
                                         n_calls_left_fn=(lambda: n_calls_max - self.remote_model.adv_get_n_calls()),
                                         n_seconds=time_left, source_step=1e-2, spherical_step=1.5e-1, give_up_calls_left=200,
                                         give_up_dist=give_up_dist)
        print("Ran BBA for {} samples.".format(self.remote_model.adv_get_n_calls() - n_calls_used_before_bia))

        return aex
