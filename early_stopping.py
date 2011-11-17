"""
An early-stopping heuristic
"""
import copy
import numpy as np
import scipy as sp

class EarlyStopping(object):
    def __init__(self, warmup, improvement_thresh=0.2, patience=2.0,
            max_iters=None):
        self.warmup = warmup
        self.improvement_thresh = improvement_thresh
        self.patience = patience
        self.max_iters = max_iters
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
        if self.cur_time >= max(
            self.warmup,
            self.best_time * self.patience):
            return True
        if self.max_iters is not None and self.cur_time >= self.max_iters:
            return True
        return False


def fit_w_early_stopping(model, es,
        train_X, train_y,
        validation_X, validation_y,
        batchsize=10,
        validation_interval=1000,
        verbose=0):

    tpos = 0
    best_model = None
    best_test_prediction = None
    best_test_errors = None

    while not es.done():
        vpos = 0
        errs = []
        test_prediction = []
        while vpos < len(validation_X):
            xi = validation_X[vpos:vpos + batchsize]
            yi = validation_y[vpos:vpos + batchsize]
            pi = model.predict(xi)
            test_prediction.extend(pi.tolist())
            assert np.all(np.isfinite(pi))
            errs.append((yi != pi).astype('float64'))
            vpos += batchsize
        test_prediction = np.array(test_prediction)

        vscore = np.mean(errs)
        # -- std dev appropriate for classification
        vscore_std = np.sqrt(vscore * (1.0 - vscore) / len(validation_X))
        es.step(vscore, vscore_std)
        if verbose:
            print ("fit_w_early_stopping: agsd weights sqrd norm: %f" % (
                model.asgd_weights ** 2).sum())
            print ("early stopper %s" % str(es))
        if best_model is None or es.cur_time == es.best_time:
            best_model = copy.deepcopy(model)
            best_test_prediction = test_prediction
            best_test_errors = errs

        # -- training loop
        for i in xrange(validation_interval):
            xi = train_X[tpos:tpos + batchsize]
            if len(xi) == 0:
                tpos = 0
                xi = train_X[tpos:tpos + batchsize]
            yi = train_y[tpos:tpos + batchsize]
            model.partial_fit(xi, yi)
            tpos += batchsize

    result = get_stats(validation_y, best_test_prediction, [-1, 1])
    result['test_errors'] = np.concatenate(best_test_errors).astype(np.int).tolist()

    return best_model, es, result


####stats

def get_stats(test_actual, test_predicted, labels):
    test_accuracy = float(100*(test_predicted == test_actual).sum() / float(len(test_predicted)))
    test_aps = []
    test_aucs = []
    if len(labels) == 2:
        labels = labels[1:]
    for label in labels:
        test_prec,test_rec = precision_and_recall(test_actual,test_predicted,label)
        test_ap = ap_from_prec_and_rec(test_prec,test_rec)
        test_aps.append(test_ap)
        test_auc = auc_from_prec_and_rec(test_prec,test_rec)
        test_aucs.append(test_auc)
    test_ap = np.array(test_aps).mean()
    test_auc = np.array(test_aucs).mean()
    return {'test_accuracy' : test_accuracy,
            'test_ap' : test_ap,
            'test_auc' : test_auc}


def precision_and_recall(actual,predicted,cls):
    c = (actual == cls)
    si = sp.argsort(-c)
    tp = sp.cumsum(sp.single(predicted[si] == cls))
    fp = sp.cumsum(sp.single(predicted[si] != cls))
    rec = tp /sp.sum(predicted == cls)
    prec = tp / (fp + tp)
    return prec,rec


def ap_from_prec_and_rec(prec,rec):
    ap = 0
    rng = sp.arange(0, 1.1, .1)
    for th in rng:
        parray = prec[rec>=th]
        if len(parray) == 0:
            p = 0
        else:
            p = parray.max()
        ap += p / rng.size
    return ap


def auc_from_prec_and_rec(prec,rec):
    #area under curve
    h = sp.diff(rec)
    auc = sp.sum(h * (prec[1:] + prec[:-1])) / 2.0
    return auc

