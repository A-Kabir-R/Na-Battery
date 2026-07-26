"""tqdm helpers, including a joblib.Parallel progress-bar patch.

`tqdm_joblib` monkey-patches joblib.parallel.BatchCompletionCallBack for the
duration of a `with` block so that a tqdm bar advances as parallel tasks finish.
Standard recipe (see tqdm docs / joblib issue #972).
"""
from __future__ import annotations

import contextlib

import joblib
from tqdm.auto import tqdm


@contextlib.contextmanager
def tqdm_joblib(tqdm_object: tqdm):
    """Context manager to make joblib.Parallel emit updates to a tqdm bar."""
    class _TqdmBatchCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _TqdmBatchCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old
        tqdm_object.close()
