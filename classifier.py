import numpy as np
import scipy as sp
import asgd  # use master branch from https://github.com/jaberg/asgd
from early_stopping import fit_w_early_stopping, EarlyStopping


def split_center_normalize(X, y,
        validset_fraction=.2,
        validset_max_examples=5000,
        inplace=False,
        min_std=1e-4,
        batchsize=1):
    n_valid = int(min(
        validset_max_examples,
        validset_fraction * X.shape[0]))

    # -- increase n_valid to a multiple of batchsize
    while n_valid % batchsize:
        n_valid += 1

    n_train = X.shape[0] - n_valid

    # -- decrease n_train to a multiple of batchsize
    while n_train % batchsize:
        n_train -= 1

    if not inplace:
        X = X.copy()

    train_features = X[:n_train]
    valid_features = X[n_train:n_train + n_valid]
    train_labels = y[:n_train]
    valid_labels = y[n_train:n_train + n_valid]

    # -- this loop is more memory efficient than numpy
    #    but not as numerically accurate as possible
    m = np.zeros(train_features.shape[1], dtype='float64')
    msq = np.zeros(train_features.shape[1], dtype='float64')
    for i in xrange(train_features.shape[0]):
        alpha = 1.0 / (i + 1)
        v = train_features[i]
        m = (alpha * v) + (1 - alpha) * m
        msq = (alpha * v * v) + (1 - alpha) * msq

    train_mean = np.asarray(m, dtype=train_features.dtype)
    train_std = np.sqrt(np.maximum(
            msq - m * m,
            min_std ** 2)).astype(train_features.dtype)

    # train features and valid features are aliased to X
    X -= train_mean
    X /= train_std

    return ((train_features, train_labels),
            (valid_features, valid_labels),
            train_mean,
            train_std)


def train_classifier_normalize(train_Xy, test_Xy, verbose=False, batchsize=10,
        step_sizes=(3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6)):

    print 'training classifier'
    train_X, train_y = train_Xy
    test_X, test_y = test_Xy

    train_mean = train_X.mean(axis=0)
    train_std = train_X.std(axis=0)

    def normalize(XX):
        return (XX - train_mean) / np.maximum(train_std, 1e-6)

    train_X = normalize(train_X)
    test_X = normalize(test_X)

    train_Xy = (train_X, train_y)
    test_Xy = (test_X, test_y)

    model, es, data = train_classifier(train_Xy, test_Xy, verbose=verbose, batchsize=batchsize,
                            step_sizes=step_sizes)

    return model, es, data, train_mean, train_std


def train_classifier(train_Xy, test_Xy, verbose=False, batchsize=10,
        step_sizes=(3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6)):
    """
    batchsize = 10                 # unit: examples
    """
    print 'training classifier'
    train_X, train_y = train_Xy
    test_X, test_y = test_Xy
    n_examples, n_features = train_X.shape

    # -- change labels to -1, +1
    if set(train_y) == set([0, 1]):
        train_y = train_y * 2 - 1
        test_y = test_y * 2 - 1

    validation_interval = 100      # unit: batches

    # -- repeat training for several learning rates
    #    take model that was best on held-out data
    results = [fit_w_early_stopping(
                model=asgd.naive_asgd.NaiveBinaryASGD(
                    n_features=n_features,
                    l2_regularization=1e-3,
                    sgd_step_size0=step_size0),
                es=EarlyStopping(warmup=50, max_iters=1000), # unit: validation intervals
                train_X=train_X,
                train_y=train_y,
                validation_X=test_X,
                validation_y=test_y,
                batchsize=batchsize,
                validation_interval=validation_interval,
                verbose=verbose
                ) for step_size0 in step_sizes]
    results.sort(cmp=lambda a, b: cmp(a[1].best_y, b[1].best_y))
    model, es, test_prediction, test_errors = results[0]
    data = get_stats(test_y, test_prediction, [1, -1])
    data['loss'] = es.best_y
    data['loss_std'] = es.best_y_std
    data['test_errors'] = np.concatenate(test_errors).astype(np.int).tolist()
    return model, es, data



def evaluate_classifier_normalize(model, test_Xy, train_mean, train_std, verbose=False, batchsize=10):
    print 'evaluating classifier'
    test_X, test_y = test_Xy
    def normalize(XX):
        return (XX - train_mean) / np.maximum(train_std, 1e-6)
    test_X = normalize(test_X)
    return evaluate_classifier(model, test_X, test_y, batchsize=batchsize, verbose=verbose)


def evaluate_classifier(model, test_X, test_y,
        batchsize=10,
        verbose=0):

    if set(test_y) == set([0, 1]):
        test_y = test_y * 2 - 1

    tpos = 0
    vpos = 0
    errs = []
    test_prediction = []
    while vpos < len(test_X):
        xi = test_X[vpos:vpos + batchsize]
        yi = test_y[vpos:vpos + batchsize]
        pi = model.predict(xi)
        test_prediction.extend(pi.tolist())
        assert np.all(np.isfinite(pi))
        errs.append((yi != pi).astype('float64'))
        vpos += batchsize
    test_prediction = np.array(test_prediction)

    vscore = np.mean(errs)
    # -- std dev appropriate for classification
    vscore_std = np.sqrt(vscore * (1.0 - vscore) / len(test_X))

    result = get_stats(test_y, test_prediction, [1, -1])
    result['loss'] = vscore
    result['loss_std'] = vscore_std
    result['test_errors'] = np.concatenate(errs).astype(np.int).tolist()

    return result


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


