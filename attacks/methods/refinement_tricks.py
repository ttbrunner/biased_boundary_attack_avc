"""
- UNUSED in current submission -

Various ideas for refining an adversarial example by modifying the pixels in certain patterns.

NOTE:
- BiasedBoundaryAttack contains these patterns, and supersedes this implementation.

TODO: Add Salt&pepper noise, or gauss, to the jitter pattern
TODO: Maybe there are other mods we can try - contrast/brightness adaptation?
"""

import numpy as np
import foolbox

from utils.util import eval_distance, eval_distance_rounded, FoolboxL2Distance
from attacks.methods import mod_boundary_attack


def refinement_loop(remote_model, X_orig, X_aex, label, is_targeted, stop_early_diff=0., sample_gen=None):
    # Runs a couple of refinements in a loop.
    # Returns a ROUNDED image and dist!

    assert hasattr(remote_model, "adv_get_n_calls"), "Please wrap the model in a RemoteModelWrapper."

    dist_orig = eval_distance_rounded(X_aex, X_orig)
    dist_pre = dist_orig
    print("At {:.2f}. Starting final reduction...".format(dist_pre))

    for i in range(5):      # TODO: repeat until call / time limit

        print("Reduction pass {}...".format(i + 1))
        print("Jittered L_inf pass:")
        n_calls_start = remote_model.adv_get_n_calls()
        X_aex = refine_jitter(remote_model, X_orig=X_orig, X_aex=X_aex, label=label, is_targeted=is_targeted, stop_early_diff=0.005, optimistic=(i == 0))
        n_calls_current = remote_model.adv_get_n_calls()
        print("Used {} calls.".format(n_calls_current - n_calls_start))

        if n_calls_current > 900:
            break

        print("L0 pass:")
        n_calls_start = n_calls_current
        X_aex = refine_pixels(remote_model, X_orig=X_orig, X_aex=X_aex, label=label, is_targeted=is_targeted, stop_early_diff=0.005, optimistic=(i == 0))
        n_calls_current = remote_model.adv_get_n_calls()
        print("Used {} calls.".format(n_calls_current - n_calls_start))

        if n_calls_current > 900:
            break

        # If the first passes weren't successful, another boundary attack won't do anything.
        dist_temp = eval_distance_rounded(X_aex, X_orig)
        if abs(dist_temp - dist_pre) < 0.0001:
            break

        print("Boundary pass:")
        n_calls_start = n_calls_current
        X_aex = refine_with_boundary_attack(remote_model, X_orig, X_aex, label, is_targeted=is_targeted,
                                            iterations=4, step=0.005, sample_gen=sample_gen)
        n_calls_current = remote_model.adv_get_n_calls()
        print("Used {} calls.".format(n_calls_current - n_calls_start))

        dist_current = eval_distance_rounded(X_aex, X_orig)
        print("Reduced by {:.2f} to {:.2f} ({:.1%}).".format(dist_pre - dist_current, dist_current, 1. - dist_current / dist_pre))
        if abs(dist_pre - dist_current) < stop_early_diff:
            break

        if n_calls_current > 900:
            break

        dist_pre = dist_current

    print("Total reduction by {:.2f} to {:.2f} ({:.1%}).".format(dist_orig - dist_pre, dist_pre, 1. - dist_pre / dist_orig))
    return np.clip(np.around(X_aex), 0, 255), dist_pre


def refine_pixels(remote_model, X_orig, X_aex, label, is_targeted, optimistic=True, stop_early_diff=0.):
    # Try to refine even further by modifying single pixels
    # Returns an UNROUNDED image!

    # Ideen:
    # - Pixel insgesamt ersetzen.
    # - Helligkeit ersetzen (Farbe gleich lassen, darauf ist er generell salient (Untersuchen?))
    # - Einzelne Bereiche modden

    X_aex = np.float32(X_aex)
    X_orig = np.float32(X_orig)
    X_orig_norm = X_orig / 255.
    img_shape = X_orig.shape

    # vis_debug = False
    # if vis_debug:
    #     # DEBUG: Plot histogram of the diffs.
    #     diff_vec = np.reshape(X_aex - X_orig, -1)
    #     import matplotlib.pyplot as plt
    #     import seaborn as sns
    #     sns.distplot(diff_vec, kde=True)
    #     plt.show(block=True)

    # Sort indices of the pixels, descending by difference to original.
    i_highest_diffs = np.argsort(np.abs(X_aex - X_orig), axis=None)[::-1]

    X_candidate_best = X_aex
    dist_best = np.linalg.norm(X_candidate_best / 255. - X_orig_norm)

    n_px_to_change = 1

    n_tries = 100
    cur_ind = 0
    for i in range(n_tries):

        X_candidate = X_candidate_best.copy()

        # Try and replace n pixels at once.
        i_pxs = i_highest_diffs[cur_ind: cur_ind + n_px_to_change]
        for i_px in i_pxs:
            i_px = np.unravel_index(i_px, img_shape)
            X_candidate[i_px] = X_orig[i_px]

        # Abort early if we don't stand to gain much.
        dist_candidate = np.linalg.norm(np.float32(X_candidate) / 255. - X_orig_norm)
        if dist_best - dist_candidate < stop_early_diff:
            break

        pred_label = np.argmax(remote_model.predictions(np.uint8(np.clip(np.around(X_candidate), 0, 255))))
        if not (pred_label == label) == is_targeted:
            # If failed: advance by one. BUT if we tried multiple, then try again and reduce the size.

            if n_px_to_change == 1:
                cur_ind += 1
            else:
                n_px_to_change = 1
            continue

        else:
            # Success: advance by one. Also be optimistic and increase the "batch" size.
            cur_ind += n_px_to_change

            if optimistic:
                n_px_to_change += 1

            if dist_candidate < dist_best:
                print("New dist - unrounded: {:.2f}, reduced by {:.1%}".format(dist_candidate, 1. - dist_candidate/dist_best))
                dist_best = dist_candidate
                X_candidate_best = X_candidate

    return X_candidate_best


def refine_jitter(remote_model, X_orig, X_aex, label, is_targeted, optimistic=True, stop_early_diff=0.):
    # Picks pixels from a jitter pattern, and slowly blends them towards the original.
    #  The rationale is: if we modify the image in a jitter pattern, it becomes very high-frequency,
    #  and the model will probably filter most of our modification.
    # Returns an UNROUNDED image!

    X_aex = np.float32(X_aex)
    X_orig = np.float32(X_orig)
    X_orig_norm = X_orig / 255.
    img_shape = X_orig.shape

    for jitter_width in [2, 7, 11, 19]:

        X_candidate_best = X_aex.copy()
        dist_best = np.linalg.norm(X_candidate_best / 255. - X_orig_norm)

        # Prepare a jitter mask with XOR (alternating). TODO: we could really improve this pattern. S&P noise, anyone?
        jitter_mask = np.empty((64, 64, 3), dtype=np.bool)
        for i in range(64):
            for j in range(64):
                jitter_mask[i, j, :] = (i % jitter_width == 0) ^ (j % jitter_width == 0)

        jitter_diff = np.zeros(img_shape, dtype=np.float32)
        jitter_diff[jitter_mask] = (X_candidate_best - X_orig)[jitter_mask]

        n_tries = 100
        eps = 2. / n_tries if optimistic else 1. / n_tries
        for i in range(n_tries):

            X_candidate = X_candidate_best - eps * jitter_diff

            # Abort early if we don't stand to gain much.
            dist_candidate = np.linalg.norm(X_candidate / 255. - X_orig_norm)
            if dist_best - dist_candidate < stop_early_diff:
                break

            pred_label = np.argmax(remote_model.predictions(np.uint8(np.clip(np.around(X_candidate), 0, 255))))
            if not (pred_label == label) == is_targeted:
                # Failure: reduce eps.
                eps /= 2.
                continue

            eps *= 1.3

            print("New dist - unrounded: {:.2f}, reduced by {:.1%}, jitter_width={}".format(dist_candidate, 1. - dist_candidate/dist_best, jitter_width))
            dist_best = dist_candidate
            X_candidate_best = X_candidate

        # Continue next it (different jitter spacing) with the best from here.
        X_aex = X_candidate_best
    return X_aex


def refine_line_binary(remote_model, X_orig, X_aex, label, is_targeted, n_tries=8):
    # Does a binary line search from x_aex to the original image.
    # Returns a ROUNDED image!

    perturbation = X_aex - X_orig

    x_best = X_aex
    factor_best = 1.
    factor = .5

    for i in range(n_tries):
        x_new = X_orig + factor * perturbation
        x_rounded = np.clip(np.round(x_new), 0, 255)
        pred_clsid = np.argmax(remote_model.predictions(x_rounded))
        if (pred_clsid == label) == is_targeted:
            x_best = x_rounded                  # This time rounding shouldn't make a difference, as we're not iterating on this
            factor_best = factor
            factor /= 2.
        else:
            factor = factor + (factor_best - factor) / 2.

    if factor_best < 1:
        dist = eval_distance_rounded(x_best, X_orig)
        print("New dist - unrounded: Managed to reduce dist to {:.2f}".format(dist))

    return x_best


def refine_with_boundary_attack(remote_model, X_orig, X_aex, label, is_targeted=False, iterations=8, step=2e-2, stop_early_diff=None,
                                sample_gen=None, normal_factor=0.0):
    # Uses the vanilla BoundaryAttack to refine an existing AEx.
    # RETURNS ROUNDED IMG AND DIST

    print_details = False

    # Clip (for strictness), but don't round.
    X_aex = np.clip(np.float32(X_aex), 0, 255)
    dist_before = eval_distance(np.round(X_aex), X_orig)

    # RemoteModelWrapper should have cached the original prediction, so it's not expensive
    original_label = np.argmax(remote_model.predictions(X_orig)) if is_targeted else label

    attack = mod_boundary_attack.BoundaryAttack()
    criterion = foolbox.criteria.TargetClass(label) if is_targeted else foolbox.criteria.Misclassification()
    adv_obj = foolbox.adversarial.Adversarial(remote_model, criterion, original_image=X_orig, original_class=original_label, distance=FoolboxL2Distance)

    x_new = attack(adv_obj, iterations=iterations, spherical_step=2*step, source_step=step, step_adaptation=1.5, max_directions=5,
                   tune_batch_size=False, starting_point=X_aex, log_every_n_steps=1 if print_details else iterations,
                   stop_early_diff=stop_early_diff, sample_gen=sample_gen, normal_factor=normal_factor)

    if x_new is not None:
        x_rounded = np.clip(np.round(x_new), 0, 255)
        dist_rounded = eval_distance(x_rounded, X_orig)
        if dist_rounded < dist_before:
            pred_clsid = np.argmax(remote_model.predictions(x_rounded))
            if (pred_clsid == label) == is_targeted:
                return x_rounded

    return X_aex
