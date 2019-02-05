from foolbox.models import Model as FoolboxModel
from timeit import default_timer
from time import sleep
import numpy as np


class TimedWrapper(FoolboxModel):
    """
    Wrapper that measures runtime of a single model query. Useful for debugging.
    """

    def __init__(self, wrapped_model, verbose=False, min_time_s=0.):
        """
        :param wrapped_model: The model to measure.
        :param verbose: If true, prints the runtime of each prediction.
        :param min_time_s: If >0, makes sure every prediction takes at least n seconds (can and should be <1.). Useful for imitating slow
                            models while debugging.
        """
        super(TimedWrapper, self).__init__(
            bounds=(0, 255),
            channel_axis=3)

        self.wrapped_model = wrapped_model
        self.n_classes = wrapped_model.num_classes()
        self.verbose = verbose
        self.min_time_s = min_time_s

        self._runtime_history_s = []

    def __enter__(self):
        assert self.wrapped_model.__enter__() == self.wrapped_model
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.wrapped_model.__exit__(exc_type, exc_value, traceback)

    def num_classes(self):
        return self.wrapped_model.num_classes()

    def batch_predictions(self, images):
        time_start = default_timer()
        preds = self.wrapped_model.batch_predictions(images)
        time_elapsed = default_timer() - time_start

        # If active, make sure the call lasts at least (n) milliseconds. This should release the GIL (just like in the online evaluation,
        #  where we are waiting via HTTP request). During this time, the attack can work in the background (prepare next candidate etc)
        if time_elapsed < self.min_time_s:
            sleep(self.min_time_s - time_elapsed)              # Releases GIL, just as the HTTP should while waiting.
            time_elapsed = default_timer() - time_start

        if len(images) == 1:
            self._runtime_history_s.append(time_elapsed)

        if self.verbose:
            if len(images) == 1:
                print("{}.batch_predictions(): processed a single image in {:.1f}ms.".format(
                    type(self.wrapped_model).__name__, time_elapsed * 1e3))
            else:
                print("{}.batch_predictions(): processed a batch of {} images in {:.1f}ms ({:.1f}ms per image).".format(
                    type(self.wrapped_model).__name__, time_elapsed * 1e3, len(images), time_elapsed * 1e3 / len(images)))

        return preds

    def get_runtime_stats(self):
        # We use these stats to get a sense of how fast the model is. If it's too slow, we must be careful that our own attack is fast,
        #  so the evaluation run doesn't get killed.
        history = np.array(self._runtime_history_s)
        return np.median(history), np.mean(history), np.std(history)

    def reset_runtime_stats(self):
        self._runtime_history_s.clear()
