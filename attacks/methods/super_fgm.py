import numpy as np
import foolbox

from attacks.methods.refinement_tricks import refine_with_boundary_attack
from models.utils.ensemble_tf_model import EnsembleTFModel
from models.utils.batch_tensorflow_model import BatchTensorflowModel
from utils.util import softmax

print_details = False


class SuperFGM:
    """
    Classic PGD transfer attack, similar to foolbox.attacks.L2BasicIterativeAttack.
    Mods:
    - Added some regional sampling to the gradient calculation, to combat stochastic defenses (see Athalye's "Obfuscated Gradients" paper).
    - Whenever a candidate is found, refines it via refinement_tricks.

    Conclusion:
    - This works sometimes, but really is inferior to our main attack (Biased Boundary Attack).
    - We can use this to provide starting points for the BBA, but they are often low quality (BBA gets stuck right away).
        If we instead initialize from a high-confidence img of the target class, often the BBA will get much further.
    """

    def __init__(self, model, substitute_model, original_image, label, is_targeted, original_label=None, sample_gen=None):
        self.model = model
        self.original_image = np.float32(np.copy(original_image))
        self.label = label
        self.is_targeted = is_targeted
        self.original_label = original_label if is_targeted else label
        self.sample_gen = sample_gen

        assert sample_gen is not None       # We need the generator for this attack.

        # Check the substitute moel. If none provided: get gradients from the original (white box).
        if substitute_model is None:
            assert isinstance(model, foolbox.models.DifferentiableModel), "This attack works only with models that provide gradients."
            self.batch_sub_model = None
        else:
            if isinstance(substitute_model, foolbox.models.TensorFlowModel):
                # We got a "vanilla" TF model. Extract the tensors so we can do batching!
                self.batch_sub_model = BatchTensorflowModel(substitute_model._images, substitute_model._batch_logits, session=substitute_model.session)
            else:
                # We got an ensemble model.... nice :)
                assert isinstance(substitute_model, EnsembleTFModel) or isinstance(substitute_model, BatchTensorflowModel)
                self.batch_sub_model = substitute_model

    def run_attack(self, do_refine=True, eps=1.5, stepsize=.75, eps_max=60., n_it_per_eps=10, bs_iterations=15,
                   random_start=True, random_start_perlin=True, noise_on_it_scale=12, n_grad_samples=8, momentum=0.0,
                   dist_early_abort=0.7, stepsize_give_up=999):

        # Here, all distances are normalized L2, just as in the competition!
        # As opposed to the Foolbox attacks, which use MSE distance!

        step_factor = stepsize / eps

        x_best = None
        dist_best = 9999.

        last_good_eps = None

        for bs_i in range(bs_iterations):
            if dist_best < dist_early_abort:
                print("Dist is good enough: {:.2f}. Aborting early to save time.".format(dist_best))
                break

            print("Eps={:.2f}, stepsize={:.2f}".format(eps, stepsize))
            x_aex, dist = self._run_per_eps(stepsize=stepsize, n_it=n_it_per_eps, eps_max=eps, abort_early=True,
                                            random_start=random_start, random_start_perlin=random_start_perlin,
                                            noise_on_it_scale=noise_on_it_scale, n_grad_samples=n_grad_samples, momentum=momentum)
            if x_aex is None:
                eps = (eps + (last_good_eps - eps) / 2.) if last_good_eps is not None else eps * 1.5
                eps = min(eps, eps_max)
                stepsize = step_factor * eps
                if stepsize > stepsize_give_up:
                    print("Stepsize got too high: {:.2f}. Giving up.".format(stepsize))
                    break

            else:
                # FGM was successful! Can we refine the image even further?
                if do_refine:
                    x_aex = refine_with_boundary_attack(self.model, self.original_image, x_aex, self.label, is_targeted=self.is_targeted,
                                                        iterations=6, step=1e-1, sample_gen=self.sample_gen)
                    dist = _l2_dist(x_aex, self.original_image)

                last_good_eps = eps             # TODO: shouldn't we set it to the refined?
                eps /= 2.
                stepsize = step_factor * eps
                if dist < dist_best:
                    print("New champion @ {:.2f}!".format(dist))
                    x_best = x_aex
                    dist_best = dist

        if do_refine and x_best is not None:
            # Assuming we have some calls left, do some more refining on the best.
            x_best = refine_with_boundary_attack(self.model, self.original_image, x_best, self.label, is_targeted=self.is_targeted,
                                                 iterations=15, step=2e-2, sample_gen=self.sample_gen)
            dist_best = _l2_dist(x_best, self.original_image)

        return x_best, dist_best

    def _run_per_eps(self, stepsize, n_it, eps_max, abort_early, random_start=True, random_start_perlin=True,
                     noise_on_it_scale=12, n_grad_samples=8, momentum=0.0):
        """
        FGM inner loop.
        :param stepsize: How far along the gradient we should move at each step.
        :param n_it: Number of steps.
        :param eps_max: If the norm of the cumulative perturbation is larger than this, clip it to this value.
        :return: UNROUNDED img and dist.
        """

        cum_gradient = np.zeros(self.original_image.shape, dtype=np.float32)

        if random_start:
            if random_start_perlin:
                noise_eps = np.random.uniform(0.01, 3 * stepsize)
                x = self.original_image + np.float32(noise_eps * self.sample_gen.get_perlin())
            else:
                x = self.original_image + np.float32(self.sample_gen.get_normal() * stepsize)
        else:
            x = np.copy(self.original_image)

        x_best = None               # WARN: this is not rounded! In the future, we might do some sidestepping.
        dist_best = 9999.

        for it in range(n_it):

            x_prev = np.copy(x)

            # Take multiple samples of the gradient and average them.
            if noise_on_it_scale > 0:

                samples = np.empty((n_grad_samples,) + self.original_image.shape, dtype=np.float32)

                for i in range(n_grad_samples):
                    # Add noise to image. TODO: Change this to EOT - countering filters and transforms
                    if random_start_perlin:
                        noise_eps = np.float32(np.random.uniform(-noise_on_it_scale, noise_on_it_scale))
                        samples[i] = x + noise_eps * self.sample_gen.get_perlin()
                    else:
                        samples[i] = x + np.float32(self.sample_gen.get_normal()) * noise_on_it_scale

                # Get gradients in a batch, if possible. This is really slow otherwise.
                if self.batch_sub_model is not None:
                    gradient_samples = self.batch_sub_model.gradient(samples, [self.label] * n_grad_samples)
                else:
                    gradient_samples = np.zeros((n_grad_samples,) + self.original_image.shape, dtype=np.float32)
                    for i in range(n_grad_samples):
                        # Get misclassification gradient.
                        gradient_samples[i] = self.model.gradient(samples[i], self.label)

                gradient = np.mean(gradient_samples, axis=0)
            else:
                if self.batch_sub_model is not None:
                    gradient = self.batch_sub_model.gradient(x[np.newaxis, :], [self.label])[0, ...]
                else:
                    gradient = self.model.gradient(x, self.label)

            if self.is_targeted:
                gradient = -gradient

            # Norm gradient to L2 distance.
            # g_norm = np.mean(np.abs(gradient))                                 # Gradient is "old school" L1 normed
            # g_norm = np.sqrt(np.vdot(gradient, gradient) / gradient.size)      # Gradient is "old school" L2 normed
            g_norm = np.linalg.norm(gradient / 255.)                             # It's the evaluation L2 norm (seems to work best)
            # print("DEBUG: gradient norm = {}".format(g_norm))
            gradient /= g_norm

            # Add previous gradients (momentum)
            cum_gradient = momentum * cum_gradient + gradient
            norm_cum_gradient = cum_gradient / np.linalg.norm(cum_gradient / 255.)

            # Add perturbation to image.
            x = x + stepsize * norm_cum_gradient

            # Normalize the (cumulative) perturbation to be of size eps. Will only scale downward, never upward.
            perturb_total = x - self.original_image
            pert_norm = _l2_norm(perturb_total)
            if pert_norm > eps_max:
                perturb_total = (perturb_total / pert_norm) * eps_max
            x = self.original_image + perturb_total

            # Round the image to uint8, making sure we remember it exactly as
            x_rounded = np.clip(np.round(x), 0, 255)
            if np.sum(np.abs(x - x_prev)) < 1e-3:
                print("WARN: Rounded/clipped img is identical to previous one!")

            # Test if adversarial.
            dist = _l2_dist(x_rounded, self.original_image)
            msg = "Trying at L2={:.3f}.".format(dist)
            pred = self.model.predictions(x_rounded)
            pred_clsid = np.argmax(pred)
            if (pred_clsid == self.label) == self.is_targeted:
                msg += " Success!"
                if dist < dist_best:
                    dist_best = dist
                    x_best = np.copy(x)
            if print_details:
                print(msg)
                pred_self = self.batch_sub_model.batch_predictions(x_rounded[np.newaxis, :])[0]
                pred_self_softmax = softmax(pred_self)
                labels = np.argsort(pred_self)[::-1]
                label_other = labels[0]
                if label_other == self.label:
                    label_other = labels[1]
                pred_self_highest = pred_self[label_other]
                pred_self_highest_softmax = pred_self_softmax[label_other]

                print("Own model reports target probability of {:.6f} (logit: {:.6f}), other is {:.6f} (logit: {:.6f})".format(
                    pred_self_softmax[self.label], pred_self[self.label], pred_self_highest_softmax, pred_self_highest))

            if abort_early and dist_best < 9999:
                break

        return x_best, dist_best


def _l2_dist(x_a, x_b):
    return np.linalg.norm(x_a / 255. - x_b / 255.)


def _l2_norm(x):
    return np.linalg.norm(x / 255.)


def extract_foolbox_(model):
    assert isinstance(model, foolbox.models.TensorFlowModel)
