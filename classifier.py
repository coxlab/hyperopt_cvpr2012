import numpy as np
import scipy as sp
import asgd  # use master branch from https://github.com/jaberg/asgd
from early_stopping import fit_w_early_stopping, EarlyStopping

def mean_and_std(X, min_std):
    # -- this loop is more memory efficient than numpy
    #    but not as numerically accurate as possible
    m = np.zeros(X.shape[1], dtype='float64')
    msq = np.zeros(X.shape[1], dtype='float64')
    for i in xrange(X.shape[0]):
        alpha = 1.0 / (i + 1)
        v = X[i]
        m = (alpha * v) + (1 - alpha) * m
        msq = (alpha * v * v) + (1 - alpha) * msq

    train_mean = np.asarray(m, dtype=X.dtype)
    train_std = np.sqrt(np.maximum(
            msq - m * m,
            min_std ** 2)).astype(X.dtype)
    return train_mean, train_std


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

    train_mean, train_std = mean_and_std(X, min_std=min_std)

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


def train_classifier(train_Xy, test_Xy,
        verbose=False,
        batchsize=10,
        validation_interval=100,
        step_sizes=(3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6),
        use_theano=False):
    """
    batchsize = 10                 # unit: examples
    validation_interval = 100      # unit: batches
    """
    train_X, train_y = train_Xy
    test_X, test_y = test_Xy
    if train_X.ndim != 2: raise TypeError()
    if test_X.ndim != 2: raise TypeError()
    n_examples, n_features = train_X.shape
    labelset = set(train_y)
    labelset.update(test_y)

    if verbose:
        print 'training classifier: n_train=%i n_test=%i n_features=%i' % (
                n_examples, len(test_X), n_features)

    assert labelset == set(range(len(labelset)))

    if labelset == set([0, 1]):
        # BINARY CLASSIFICATION
        # -- change labels to -1, +1
        train_y = train_y * 2 - 1
        test_y = test_y * 2 - 1
        def model_fn(step_size0):
            return asgd.naive_asgd.NaiveBinaryASGD(
                    n_features=n_features,
                    l2_regularization=1e-3,
                    sgd_step_size0=step_size0)

        if use_theano:
            print "train_classifier not using theano for binary classif."
    else:
        # MULTI-CLASS CLASSIFICATION
        if use_theano:
            def model_fn(step_size0):
                return TheanoSGD(
                        n_features=n_features,
                        n_classes=len(labelset),
                        batchsize=batchsize,
                        l1_regularization=0,
                        l2_regularization=1e-6,
                        learnrate=step_size0,
                        )
        else:
            def model_fn(step_size0):
                return asgd.naive_asgd.NaiveMulticlassASGD(
                        n_features=n_features,
                        n_classes=len(labelset),
                        l2_regularization=1e-3,
                        sgd_step_size0=step_size0)

    # -- repeat training for several learning rates
    #    take model that was best on held-out data
    results = [fit_w_early_stopping(
                model_fn(step_size0),
                es=EarlyStopping(warmup=50, max_iters=1000), # unit: validation intervals
                train_X=train_X,
                train_y=train_y,
                validation_X=test_X,
                validation_y=test_y,
                batchsize=batchsize,
                validation_interval=validation_interval,
                verbose=verbose
                )
            for step_size0 in step_sizes]

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
    return evaluate_classifier(model, (test_X, test_y), batchsize=batchsize, verbose=verbose)


def evaluate_classifier(model, test_Xy,
        batchsize=10,
        verbose=0):

    test_X, test_y = test_Xy
    
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


# Example of a BaseEstimator implemented with Theano.
# This could use the GPU, except that
# a) linear regression isn't really worth it, and
# b) the multi_hinge_margin Op is only implemented for the CPU.
#
import theano
from theano import tensor
from theano.compile import shared
from theano.tensor import nnet
from theano import gof
from theano.tensor import Apply
from theano import tensor

class MultiHingeMargin(gof.Op):
    """
    This is a hinge loss function for multiclass predictions.

    For each vector X[i] and label index yidx[i],
    output z[i] = 1 - margin

    where margin is the difference between X[i, yidx[i]] and the maximum other element of X[i].
    """
    default_output = 0
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return tensor.hashtype(self)
    def __str__(self):
        return self.__class__.__name__
    def make_node(self, X, yidx):
        X_ = tensor.as_tensor_variable(X)
        yidx_ = tensor.as_tensor_variable(yidx)
        if X_.type.ndim != 2:
            raise TypeError('X must be matrix')
        if yidx.type.ndim != 1:
            raise TypeError('yidx must be vector')
        if 'int' not in str(yidx.type.dtype):
            raise TypeError("yidx must be integers, it's a vector of class labels")
        hinge_loss = tensor.vector(dtype=X.dtype)
        winners = X.type()
        return Apply(self, [X_, yidx_], [hinge_loss, winners])
    def perform(self, node, input_storage, out):
        X, yidx = input_storage
        toplabel = X.shape[1]-1
        out[0][0] = z = np.zeros_like(X[:,0])
        out[1][0] = w = np.zeros_like(X)
        for i,Xi in enumerate(X):
            yi = yidx[i]
            if yi == 0:
                next_best = Xi[1:].argmax()+1
            elif yi==toplabel:
                next_best = Xi[:toplabel].argmax()
            else:
                next_best0 = Xi[:yi].argmax()
                next_best1 = Xi[yi+1:].argmax()+yi+1
                next_best = next_best0 if Xi[next_best0]>Xi[next_best1] else next_best1
            margin = Xi[yi] - Xi[next_best]
            if margin < 1:
                z[i] = 1 - margin
                w[i,yi] = -1
                w[i,next_best] = 1
    def grad(self, inputs, g_outs):
        z = self(*inputs)
        w = z.owner.outputs[1]
        gz, gw = g_outs
        if gw is not None:
            raise NotImplementedError()
        gX = gz.dimshuffle(0,'x') * w
        return [gX, None]
    def c_code_cache_version(self):
        return (1,)
    def c_code(self, node, name, (X, y_idx), (z,w), sub):
        return '''
        if ((%(X)s->descr->type_num != PyArray_DOUBLE) && (%(X)s->descr->type_num != PyArray_FLOAT))
        {
            PyErr_SetString(PyExc_TypeError, "types should be float or float64");
            %(fail)s;
        }
        if ((%(y_idx)s->descr->type_num != PyArray_INT64)
            && (%(y_idx)s->descr->type_num != PyArray_INT32)
            && (%(y_idx)s->descr->type_num != PyArray_INT16)
            && (%(y_idx)s->descr->type_num != PyArray_INT8))
        {
            PyErr_SetString(PyExc_TypeError, "y_idx not int8, int16, int32, or int64");
            %(fail)s;
        }
        if ((%(X)s->nd != 2)
            || (%(y_idx)s->nd != 1))
        {
            PyErr_SetString(PyExc_ValueError, "rank error");
            %(fail)s;
        }
        if (%(X)s->dimensions[0] != %(y_idx)s->dimensions[0])
        {
            PyErr_SetString(PyExc_ValueError, "dy.shape[0] != sm.shape[0]");
            %(fail)s;
        }
        if ((NULL == %(z)s)
            || (%(z)s->dimensions[0] != %(X)s->dimensions[0]))
        {
            Py_XDECREF(%(z)s);
            %(z)s = (PyArrayObject*) PyArray_SimpleNew(1, PyArray_DIMS(%(X)s),
                                                        type_num_%(X)s);
            if (!%(z)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc dx output");
                %(fail)s;
            }
        }
        if ((NULL == %(w)s)
            || (%(w)s->dimensions[0] != %(X)s->dimensions[0])
            || (%(w)s->dimensions[1] != %(X)s->dimensions[1]))
        {
            Py_XDECREF(%(w)s);
            %(w)s = (PyArrayObject*) PyArray_SimpleNew(2, PyArray_DIMS(%(X)s),
                                                        type_num_%(X)s);
            if (!%(w)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc dx output");
                %(fail)s;
            }
        }

        for (size_t i = 0; i < %(X)s->dimensions[0]; ++i)
        {
            const dtype_%(X)s* __restrict__ X_i = (dtype_%(X)s*) (%(X)s->data + %(X)s->strides[0] * i);
            npy_intp SX = %(X)s->strides[1]/sizeof(dtype_%(X)s);

            dtype_%(w)s* __restrict__ w_i = (dtype_%(w)s*) (%(w)s->data + %(w)s->strides[0] * i);
            npy_intp Sw = %(w)s->strides[1]/sizeof(dtype_%(w)s);

            const dtype_%(y_idx)s y_i = ((dtype_%(y_idx)s*)(%(y_idx)s->data + %(y_idx)s->strides[0] * i))[0];

            dtype_%(X)s X_i_max = X_i[0];
            dtype_%(X)s X_at_y_i = X_i[0];
            size_t X_i_argmax = 0;
            size_t j = 1;
            w_i[0] = 0;

            if (y_i == 0)
            {
                X_i_max = X_i[SX];
                X_i_argmax = 1;
                w_i[Sw] = 0;
            }
            for (; j < %(X)s->dimensions[1]; ++j)
            {
                dtype_%(X)s  X_ij = X_i[j*SX];
                if (j == y_i)
                {
                    X_at_y_i = X_ij;
                }
                else if (X_ij > X_i_max)
                {
                    X_i_max = X_ij;
                    X_i_argmax = j;
                }
                w_i[j*Sw] = 0;
            }
            if (0 < 1 - X_at_y_i + X_i_max)
            {
                ((dtype_%(z)s*)(%(z)s->data + %(z)s->strides[0] * i))[0]
                    = 1 - X_at_y_i + X_i_max;
                w_i[y_i*Sw] = -1;
                w_i[X_i_argmax*Sw] = 1;
            }
        }
        ''' % dict(locals(), **sub)
multi_hinge_margin = MultiHingeMargin()

def sgd_updates(params, grads, stepsizes):
    """Return a list of (pairs) that can be used as updates in theano.function to implement
    stochastic gradient descent.

    :param params: variables to adjust in order to minimize some cost
    :type params: a list of variables (theano.function will require shared variables)
    :param grads: the gradient on each param (with respect to some cost)
    :type grads: list of theano expressions
    :param stepsizes: step by this amount times the negative gradient on each iteration
    :type stepsizes: [symbolic] scalar or list of one [symbolic] scalar per param
    """
    try:
        iter(stepsizes)
    except Exception:
        stepsizes = [stepsizes for p in params]
    if len(params) != len(grads):
        raise ValueError('params and grads have different lens')
    updates = [(p, p - step * gp) for (step, p, gp) in zip(stepsizes, params, grads)]
    return updates

class LogisticRegression(object):
    def __init__(self, input, w, b, params=[]):
        self.output=nnet.softmax(theano.dot(input, w)+b)
        self.l1=abs(w).sum()
        self.l2_sqr = (w**2).sum()
        self.argmax=theano.tensor.argmax(theano.dot(input, w)+b, axis=input.ndim-1)
        self.input = input
        self.w = w
        self.b = b
        self.params = params

    @classmethod
    def new(cls, input, n_in, n_out, dtype=None, name=None):
        if dtype is None:
            dtype = input.dtype
        if name is None:
            name = cls.__name__
        w = shared(np.zeros((n_in, n_out), dtype=dtype), name='%s.w'%name)
        b = shared(np.zeros((n_out,), dtype=dtype), name='%s.b'%name)
        return cls(input, w, b, params=[w,b])


    def nll(self, target):
        """Return the negative log-likelihood of the prediction of this model under a given
        target distribution.  Passing symbolic integers here means 1-hot.
        WRITEME
        """
        return nnet.categorical_crossentropy(self.output, target)

    def errors(self, target):
        """Return a vector of 0s and 1s, with 1s on every line that was mis-classified.
        """
        if target.ndim != self.argmax.ndim:
            raise TypeError('target should have the same shape as self.argmax', ('target', target.type,
                'argmax', self.argmax.type))
        if target.dtype.startswith('int'):
            return theano.tensor.neq(self.argmax, target)
        else:
            raise NotImplementedError()

class TheanoSGD(object):
    def __init__(self,
            n_features,
            n_classes,
            batchsize = 100,
            learnrate = 0.005,
            l1_regularization = 0.0,
            l2_regularization = 0.0,
            anneal_epoch=20,
            epoch_len = 10000,       # examples
            loss_fn='hinge',
            dtype='float32'
            ):
        # add arguments to class
        self.__dict__.update(locals())
        del self.self

        x_i = tensor.matrix(dtype=dtype)
        y_i = tensor.vector(dtype='int64')
        lr = tensor.scalar(dtype=dtype)

        feature_logreg = LogisticRegression.new(x_i,
                n_in = n_features, n_out=self.n_classes,
                dtype=x_i.dtype)

        if self.loss_fn=='log':
            traincost = feature_logreg.nll(y_i).mean()
        elif self.loss_fn=='hinge':
            raw_output = (tensor.dot(feature_logreg.input, feature_logreg.w)
                    + feature_logreg.b)
            traincost = multi_hinge_margin(raw_output, y_i).mean()
        else:
            raise NotImplementedError(self.loss_fn)

        traincost = traincost + abs(feature_logreg.w).sum() * self.l1_regularization
        traincost = traincost + (feature_logreg.w**2).sum() * self.l2_regularization

        self._train_fn = theano.function([x_i, y_i, lr],
                [feature_logreg.nll(y_i).mean(),
                    feature_logreg.errors(y_i).mean()],
                updates=sgd_updates(
                    params=feature_logreg.params,
                    grads=tensor.grad(traincost, feature_logreg.params),
                    stepsizes=[lr, lr / n_classes]))

        self._test_fn = theano.function([x_i, y_i],
                feature_logreg.errors(y_i))

        self._predict_fn = theano.function([x_i], feature_logreg.argmax)
        self.n_examples_fitted = 0

    def partial_fit(self, X, y):
        assert len(X) == len(y)

        epoch = self.n_examples_fitted / self.epoch_len

        # Marc'Aurelio, you crazy!!
        # the division by batchsize is done in the cost function
        e_lr = np.float32(self.learnrate /
                max(1.0, np.floor(
                    max(1.0,
                        (epoch + 1) /
                            float(self.anneal_epoch))
                    -2)))

        nll, l01 = self._train_fn(X, y, e_lr)

        self.n_examples_fitted += len(X)

    def predict(self, X):
        return self._predict_fn(X)




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


