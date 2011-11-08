"""
An early-stopping heuristic
"""
import copy
import numpy as np

class EarlyStopping(object):
    def __init__(self, warmup, improvement_thresh=0.2, patience=2.0):
        self.warmup = warmup
        self.improvement_thresh = improvement_thresh
        self.patience = patience
        self.cur_time = 0
        self.best_time = -1
        self.best_y = float('inf')
        self.best_y_std = 0
        self.cur_y = None
        self.cur_y_std = None

    def __str__(self):
        return ('EarlyStopping cur_time=%i cur_y=%f'
                ' best_time=%i best_y=%f +- %f') % (
                self.cur_time, self.cur_y,
                self.best_time, self.best_y, self.best_y_std
                )

    def step(self, y, y_std):
        if y_std < 0:
            raise ValueError('negative y_std', y_std)

        self.cur_time += 1
        self.cur_y = y
        self.cur_y_std = y_std

        if y < (self.best_y - self.improvement_thresh * self.best_y_std):
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
        vpos = 0
        errs = []
        while vpos < len(validation_X):
            xi = validation_X[vpos:vpos + batchsize]
            yi = validation_y[vpos:vpos + batchsize]
            pi = model.predict(xi)
            errs.append((yi != pi).astype('float64'))
            vpos += batchsize

        vscore = np.mean(errs)
        # -- std dev appropriate for classification
        vscore_std = np.sqrt(vscore * (1.0 - vscore) / len(validation_X))
        es.step(vscore, vscore_std)
        print es, 'margin_avg', model.margin_avg
        print (model.asgd_weights ** 2).sum()
        if es.cur_time > es.warmup and es.cur_time == es.best_time:
            best_model = copy.deepcopy(model)

        # -- training loop
        for i in xrange(validation_interval):
            xi = train_X[tpos:tpos + batchsize]
            if len(xi) == 0:
                tpos = 0
                xi = train_X[tpos:tpos + batchsize]
            yi = train_y[tpos:tpos + batchsize]
            model.partial_fit(xi, yi)
            tpos += batchsize


    return model if best_model is None else best_model

