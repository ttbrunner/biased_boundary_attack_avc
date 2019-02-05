import numpy as np
from timeit import default_timer

from adversarial_vision_challenge import load_model
from adversarial_vision_challenge import read_images
from adversarial_vision_challenge import store_adversarial
from adversarial_vision_challenge import attack_complete

from utils import util
from utils.sampling.sample_generator import SampleGenerator
from wrappers.remote_model_wrapper import RemoteModelWrapper
from wrappers.timed_wrapper import TimedWrapper
from attacks.multi_boundary_attack import MultiBoundaryAttack
from utils.dataset import load_dataset


def main():

    n_classes = 200
    img_shape = (64, 64, 3)

    timed_wrapper = TimedWrapper(load_model())                                  # Measure model prediction runtime.
    remote_wrapper = RemoteModelWrapper(timed_wrapper, do_hash=True)            # Remember best adv. ex.

    with SampleGenerator(shape=img_shape, n_threads=1, queue_lengths=100) as sample_gen:

        X_train, y_train, X_val, y_val = load_dataset("/path/to/tiny/imagenet", ds_cache_path='tiny_imagenet_cached.npz')

        with MultiBoundaryAttack(model=remote_wrapper,
                                 X_known=np.vstack([X_train, X_val]),
                                 y_known=np.concatenate([y_train, y_val]),
                                 n_classes=n_classes,
                                 sample_gen=sample_gen,
                                 cached_conf=None) as attack:

            model_mean_query_time_history = []
            time_max = 89                           # As allowed in the rules (batch of 10 in 900 seconds)
            time_bonus = 0                          # Bonus to account for unfair models (see below)

            i = 0
            for (file_name, image, label) in read_images():

                time_start = default_timer()

                # Time calculation: 90 seconds per image are allowed. Models are allowed to use (40ms*1000calls) = 40s.
                # This leaves 50 seconds for the attacker.
                #
                # But if the model is MUCH slower than allowed, then the attacker has less time and can't finish.
                # To balance the scales, we detect this, and allow ourselves to use up some extra seconds.
                # If we don't do this (and hard-abort at 90 seconds), attacks that don't count time would have an advantage vs us.
                if i % 5 == 0 and len(model_mean_query_time_history) > 3:
                    avg_model_time = np.mean(model_mean_query_time_history)
                    if avg_model_time > 55e-3:
                        time_left_for_attacker = 89 - (1000 * avg_model_time)
                        time_bonus = min(55 - time_left_for_attacker, 50)
                        print("Model is slower than allowed (would leave only {:.1f} seconds for the attacker). "
                              "Will now use up to {:.1f} additional seconds per image.".format(time_left_for_attacker, time_bonus))
                    elif time_bonus > 0:
                        time_bonus = 0
                        print("Model speed seems OK now. Reverting to the 90s time limit.")

                print("Image {}:".format(i))
                image = np.float32(image)

                remote_wrapper.adv_set_target(orig_img=image, is_targeted=True, label=label)
                attack.run_attack(image=image, label=label, is_targeted=True,
                                  start_with_fgm=True, fgm_acceptable_dist=10, time_max=time_max + time_bonus)
                safe_adversarial = remote_wrapper.adv_get_best_img()

                if safe_adversarial is None:
                    safe_adversarial = np.uint8(image)
                    print("Couldn't find an adversarial! This sucks!")
                else:
                    dist = util.eval_distance(image, safe_adversarial)
                    print("Final distance: {}".format(dist))

                # Save model query time stats.
                rt_median, rt_mean, rt_std = timed_wrapper.get_runtime_stats()
                print("Response time of model: median={:.1f}ms, mean={:.1f}ms, std={:.1f}ms".format(
                    rt_median * 1e3, rt_mean * 1e3, rt_std * 1e3))
                timed_wrapper.reset_runtime_stats()
                if remote_wrapper.adv_get_n_calls() > 100:
                    model_mean_query_time_history.append(rt_mean)

                time_elapsed_s = default_timer() - time_start
                print("Queried the model {} times.".format(remote_wrapper.adv_get_n_calls()))
                print("Attack for this image took {} seconds.".format(time_elapsed_s))
                print()

                store_adversarial(file_name, safe_adversarial)
                i += 1

            attack_complete()


if __name__ == '__main__':
    main()
