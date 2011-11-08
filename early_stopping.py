"""
An early-stopping heuristic
"""
import copy
import numpy as np

class early_stopping(object):
    def __init__(self, warmup, improvement_thresh=0.5, patience=2.0):
        self.warmup = warmup
        self.improvement_thresh = improvement_thresh
        self.patience = patience
        self.cur_time = 0
        self.best_time = -1
        self.best_y = float('inf')
        self.best_y_std = 0
        self.cur_y = None
        self.cur_y_std = None

    def step(self, y, y_std):
        if y_std < 0:
            raise ValueError('negative y_std', y_std)

        self.cur_time += 1
        self.cur_y = y
        self.cur_y_std = y_std

        if y < self.best_y - self.improvement_thresh * self.best_y_std:
            self.best_time = self.cur_time
            self.best_y = y
            self.best_y_std = y_std

    def done(self):
        return self.cur_time >= max(
            self.warmup,
            self.best_time * self.patience)


def fit_w_early_stopping(model, es,
        train_X, train_y,
        validation_X, validation_y,
        batchsize=10,
        validation_interval=1000):

    tpos = 0
    best_model = None

    while not es.done():
        for i in xrange(validation_interval):
            xi = train_X[tpos:tpos + batchsize]
            if len(xi) == 0:
                tpos = 0
                xi = train_X[tpos:tpos + batchsize]
            yi = train_y[tpos:tpos + batchsize]
            model.partial_fit(xi, yi)
            tpos += batchsize

        vpos = 0
        errs = []
        while vpos < len(validation_X):
            xi = validation_X[vpos:vpos + batchsize]
            yi = validation_y[vpos:vpos + batchsize]
            pi = model.predict(xi)
            errs.append(yi != pi)

        vscore = np.mean(errs)
        vscore_std = vscore * (1.0 - vscore) / np.sqrt(len(validation_X))
        es.step(vscore, vscore_std)
        if es.cur_time > es.warmup and es.cur_time == es.best_time:
            best_model = copy.deepcopy(model)

    return model if best_model is None else best_model
