import numpy as np

import early_stopping

def test_monotonic_decrease():
    rng = np.random.RandomState(2345)
    es = early_stopping.early_stopping(warmup=10)
    best_ys = []
    while not es.done():
        y = rng.rand()
        es.step(y, .01)
        best_ys.append(es.best_y)

    assert len(best_ys) == 54
    assert best_ys == list(reversed(sorted(best_ys)))


def test_warmup():
    # using same seed as above that quit after 54 iters,
    # this test raises warmup to 100, forcing more iters.
    rng = np.random.RandomState(2345)
    es = early_stopping.early_stopping(warmup=100)
    best_ys = []
    while not es.done():
        y = rng.rand()
        es.step(y, .01)
        best_ys.append(es.best_y)

    assert len(best_ys) > 54, len(best_ys)
    assert best_ys == list(reversed(sorted(best_ys)))
