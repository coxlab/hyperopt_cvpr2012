import sys
import time
import os
import copy
import itertools
import tempfile
import os.path as path
import hashlib
import cPickle

import Image
import numpy as np
from bson import BSON, SON

import theano
import theano.tensor as tensor
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from pythor3.model.slm.plugins.passthrough.passthrough import (
        SequentialLayeredModelPassthrough,
        )
from pythor3.model import SequentialLayeredModel
from pythor3.operation import fbcorr_
from pythor3.operation import lnorm_

import skdata.larray
import skdata.utils
import skdata.lfw
import hyperopt.genson_bandits as gb
from hyperopt.mongoexp import MongoJobs, MongoExperiment, as_mongo_str

# importing here somehow makes unpickling work, despite circular import in
# hyperopt
import hyperopt.theano_bandit_algos

from utils import TooLongException
from theano_slm import TheanoSLM, interpret_model, FeatureExtractor
from classifier import train_classifier, split_center_normalize, mean_and_std
from lfw import get_relevant_images


try:
    import sge_utils
except ImportError:
    pass
import cvpr_params
import comparisons as comp_module


def son_to_py(son):
    """ turn son keys (unicode) into str
    """
    if isinstance(son, SON):
        return dict([(str(k), son_to_py(v)) for k, v in son.items()])
    elif isinstance(son, list):
        return [son_to_py(s) for s in son]
    elif isinstance(son, basestring):
        return str(son)
    else:
        return son


class Bandit1(gb.GensonBandit):
    """
    This Bandit has the same evaluate function as LFWBandit,
    but the template is setup for more efficient search.
    """
    def __init__(self):
        from cvpr_params import (
                choice, uniform, gaussian, lognormal, ref, null, qlognormal)
        lnorm = {'kwargs':{'inker_shape' : choice([(3,3),(5,5),(7,7),(9,9)]),
                 'outker_shape' : ref('this','inker_shape'),
                 'remove_mean' : choice([0,1]),
                 'stretch' : lognormal(0, 1),
                 'threshold' : lognormal(0, 1)
                 }}
        lpool = dict(
                kwargs=dict(
                    stride=2,
                    ker_shape=choice([(3,3),(5,5),(7,7),(9,9)]),
                    order=choice([1, 2, 10, uniform(1, 10)])))
        activ =  {'min_out' : choice([null,0]), 'max_out' : choice([1,null])}

        filter1 = dict(
                initialize=dict(
                    filter_shape=choice([(3,3),(5,5),(7,7),(9,9)]),
                    n_filters=qlognormal(np.log(32), 1, round=16),
                    generate=(
                        'random:uniform',
                        {'rseed': choice([11, 12, 13, 14, 15])})),
                kwargs=activ)

        filter2 = dict(
                initialize=dict(
                    filter_shape=choice([(3, 3), (5, 5), (7, 7), (9, 9)]),
                    n_filters=qlognormal(np.log(32), 1, round=16),
                    generate=(
                        'random:uniform',
                        {'rseed': choice([21, 22, 23, 24, 25])})),
                kwargs=activ)

        filter3 = dict(
                initialize=dict(
                    filter_shape=choice([(3, 3), (5, 5), (7, 7), (9, 9)]),
                    n_filters=qlognormal(np.log(32), 1, round=16),
                    generate=(
                        'random:uniform',
                        {'rseed': choice([31, 32, 33, 34, 35])})),
                kwargs=activ)

        layers = [[('lnorm', lnorm)],
                  [('fbcorr', filter1), ('lpool', lpool), ('lnorm', lnorm)],
                  [('fbcorr', filter2), ('lpool', lpool), ('lnorm', lnorm)],
                  [('fbcorr', filter3), ('lpool', lpool), ('lnorm', lnorm)]]

        comparison = ['mult', 'absdiff', 'sqrtabsdiff', 'sqdiff']

        config = {'desc' : layers,
                'comparison' : comparison,
                }
        source_string = repr(config).replace("'",'"')
        gb.GensonBandit.__init__(self, source_string=source_string)

    @classmethod
    def evaluate(cls, config, ctrl, use_theano=True):
        raise NotImplementedError()


class ImgLoaderResizer(object):
    """ Load 250x250 greyscale images, return normalized 200x200 float32 ones.
    """
    def __init__(self, shape=None, ndim=None, dtype='float32', mode=None):
        assert shape == (200, 200)
        assert dtype == 'float32'
        self._shape = shape
        if ndim is None:
            self._ndim = None if (shape is None) else len(shape)
        else:
            self._ndim = ndim
        self._dtype = dtype
        self.mode = mode

    def rval_getattr(self, attr, objs):
        if attr == 'shape' and self._shape is not None:
            return self._shape
        if attr == 'ndim' and self._ndim is not None:
            return self._ndim
        if attr == 'dtype':
            return self._dtype
        raise AttributeError(attr)

    def __call__(self, file_path):
        im = Image.open(file_path)
        im = im.resize((200, 200), Image.ANTIALIAS)
        rval = np.asarray(im, 'float32')
        rval -= rval.mean()
        rval /= max(rval.std(), 1e-3)
        assert rval.shape == (200, 200)
        return rval


class PairFeatures(object):
    def __init__(self, dataset, features, comparison, filename_prefix):
        self.dataset = dataset
        self.features = features
        self.comparison = comparison
        self.filename_prefix = filename_prefix
        self.idx_of_path = idx_of_path = {}
        for i, dct in enumerate(dataset.meta):
            idx_of_path[dataset.image_path(dct)] = i

    def compute_pairs_features(self, lpaths, rpaths):
        arr_shape = (len(lpaths),
                self.comparison.get_num_features(self.features.shape))
        arr = np.empty(arr_shape, dtype=self.features.dtype)

        # -- load features into memory if possible
        #    for random access
        idxable_features = np.array(self.features)

        for (i, (lpath, rpath)) in enumerate(zip(lpaths, rpaths)):
            arr[i] = self.comparison(
                    idxable_features[self.idx_of_path[lpath]],
                    idxable_features[self.idx_of_path[rpath]])
            if i % 100 == 0:
                print('get_pair_fp  %i / %i' % (i, len(arr)))
        return arr

    def compute_pairs_features_fliplr(self, lpaths, rpaths, labels):
        arr_shape = (len(lpaths) * 4,
                self.comparison.get_num_features(self.features.shape))
        arr = np.empty(arr_shape, dtype=self.features.dtype)
        arr_labels = np.empty(arr_shape[0], dtype='int')

        # -- load features into memory if possible
        #    for random access
        idxable_features = np.array(self.features)

        for (ii, (lpath, rpath)) in enumerate(zip(lpaths, rpaths)):
            lfeat = idxable_features[self.idx_of_path[lpath]]
            rfeat = idxable_features[self.idx_of_path[rpath]]
            arr[4 * ii + 0] = self.comparison(lfeat, rfeat)
            arr[4 * ii + 1] = self.comparison(lfeat[:, ::-1], rfeat)
            arr[4 * ii + 2] = self.comparison(lfeat, rfeat[:, ::-1])
            arr[4 * ii + 3] = self.comparison(lfeat[:, ::-1], rfeat[:, ::-1])
            arr_labels[4 * ii: 4 * ii + 4] = labels[ii]
            if ii % 100 == 0:
                print('get_pair_fp  %i / %i' % (ii * 4, len(arr)))
        return arr, arr_labels

    def get_paths(self, splits):
        return map(np.concatenate,
                zip(*[self.dataset.raw_verification_task(split=split)
                    for split in splits]))

    def match_task(self, splits, flip_lr=False):
        if isinstance(splits, str):
            splits = [splits]

        lpaths, rpaths, labels = self.get_paths(splits)

        if flip_lr:
            return self.compute_pairs_features_fliplr(lpaths, rpaths, labels)
        else:
            arr = self.compute_pairs_features(lpaths, rpaths)
            return arr, labels


def main_result_from_trial():
    _, cmd, bandit_str, algo_str, trials_idx = sys.argv
    mj = MongoJobs.new_from_connection_str(
            as_mongo_str('localhost/hyperopt/jobs'))
    mexp = MongoExperiment.from_exp_key(mj,
            exp_key = '%s/%s' % (bandit_str, algo_str))
    mexp.refresh_trials_results()
    print mexp.results[int(trials_idx)]


def main_features_from_dan():
    _, cmd, idx, filename = sys.argv
    idx = int(idx)

    batchsize=4
    dataset = skdata.lfw.Aligned()
    Xr, yr = dataset.raw_classification_task()
    Xr = np.array(Xr)
    X = skdata.larray.lmap(
                ImgLoaderResizer(
                    shape=(200, 200),  # lfw-specific
                    dtype='float32'),
                Xr)
    print('extracting from trial %i' % idx)
    desc = cPickle.load(open('/home/bergstra/Downloads/thingy.pkl'))[idx]['spec']['desc']
    interpret_model(desc)
    if X.ndim == 3:
        slm = TheanoSLM(
                in_shape=(batchsize,) + X.shape[1:] + (1,),
                description=desc)
    elif X.ndim == 4:
        slm = TheanoSLM(
                in_shape=(batchsize,) + X.shape[1:],
                description=desc)
    else:
        raise NotImplementedError()
    slm.get_theano_fn()  # -- pre-compile the feature extractor

    extractor = FeatureExtractor(X, slm,
            filename=filename,
            batchsize=batchsize,
            TEST=False)
    extractor.compute_features(use_memmap=True)


def main_features_from_trial():
    _, cmd, bandit_str, algo_str, trials_idx, filename = sys.argv
    trials_idx = int(trials_idx)

    mj = MongoJobs.new_from_connection_str(
            as_mongo_str('localhost/hyperopt/jobs'))
    mexp = MongoExperiment.from_exp_key(mj,
            exp_key = '%s/%s' % (bandit_str, algo_str))
    mexp.refresh_trials_results()

    batchsize=4
    dataset = skdata.lfw.Aligned()
    Xr, yr = dataset.raw_classification_task()
    Xr = np.array(Xr)
    X = skdata.larray.lmap(
                ImgLoaderResizer(
                    shape=(200, 200),  # lfw-specific
                    dtype='float32'),
                Xr)
    print('extracting from trial %i' % trials_idx)
    desc = copy.deepcopy(son_to_py(mexp.trials[trials_idx]['desc']))
    interpret_model(desc)
    if X.ndim == 3:
        slm = TheanoSLM(
                in_shape=(batchsize,) + X.shape[1:] + (1,),
                description=desc)
    elif X.ndim == 4:
        slm = TheanoSLM(
                in_shape=(batchsize,) + X.shape[1:],
                description=desc)
    else:
        raise NotImplementedError()
    slm.get_theano_fn()  # -- pre-compile the feature extractor

    extractor = FeatureExtractor(X, slm,
            filename=filename,
            batchsize=batchsize,
            TEST=False)
    extractor.compute_features(use_memmap=True)


def main_classify_features():
    _, cmd, in_filename, comparison, flip_lr, train_test, fold, outfile = sys.argv

    in_dtype, in_shape = cPickle.load(open(in_filename + '.info'))
    features = np.memmap(in_filename, dtype=in_dtype, mode='r', shape=in_shape)
    print('loaded features of shape %s' % str(features.shape))
    dataset = skdata.lfw.Aligned()

    flip_lr = bool(int(flip_lr))

    pf = PairFeatures(dataset, features,
            comparison=getattr(comp_module, comparison),
            filename_prefix='pairs')

    if train_test == 'train':
        print "OPTIMIZING TEST SET PERFORMANCE OF OFFICIAL DEV SPLIT"
        train_Xy = pf.match_task('DevTrain', flip_lr=flip_lr)
        test_Xy = pf.match_task('DevTest', flip_lr=False)

        train_X, train_y = train_Xy
        test_X, test_y = test_Xy

        m = np.mean(train_X, axis=0)
        s = np.std(train_X, axis=0)
        train_X = (train_X - m) / s
        test_X = (test_X - m) / s

        if 1:
            np.random.RandomState(123).shuffle(train_X)
            np.random.RandomState(123).shuffle(train_y)

        model, earlystopper, data = train_classifier(
                (train_X, train_y),
                (test_X, test_y), verbose=True)
        print 'best y', earlystopper.best_y
        print 'best time', earlystopper.best_time

    else:
        assert train_test == 'test'
        if fold is None:
            folds = range(10)
        else:
            folds = [int(fold)]

        results = {}
        for i in folds:
            print('evaluating fold %d ....' % i)
            test_split = 'fold_' + str(i)
            v_ind = (i + 1) % 10
            validate_split = 'fold_' + str(v_ind)
            inds = range(10)
            inds.remove(i)
            inds.remove(v_ind)
            train_splits = ['fold_' + str(ind) for ind in inds]
            train_Xy = pf.match_task(train_splits, flip_lr=flip_lr)
            validate_Xy = pf.match_task(validate_split, flip_lr=False)
            test_Xy = pf.match_task(test_split, flip_lr=False)

            train_X, train_y = train_Xy
            validate_X, validate_y = validate_Xy
            test_X, test_y = test_Xy

            m = np.mean(train_X, axis=0)
            s = np.std(train_X, axis=0)
            train_X = (train_X - m) / s
            validate_X = (validate_X - m) / s
            test_X = (test_X - m) / s

            model, earlystopper, data = train_classifier((train_X, train_y),
                                                   (validate_X, validate_y),
                                                   verbose=True)
            print 'best validation_y', earlystopper.best_y
            print 'best validation time', earlystopper.best_time
            result = evaluate_classifier(model, (test_X, test_y), verbose=True)
            print 'best result', result['loss']
            results['split_' + str(i)] = result

        if outfile is not None:
            import cPickle
            fp = open(outfile,'w')
            cPickle.dump(results, fp)
            fp.close()


def _main_featureextract_helper(config, filename, splits):
    batchsize=4
    dataset = skdata.lfw.Aligned()
    Xr, yr = dataset.raw_classification_task()
    X, y, Xpaths = get_relevant_images(dataset,
            splits=splits,
            dtype='float32')
    print('extracting for config:')
    print config
    desc = config['desc']
    interpret_model(desc)
    if X.ndim == 3:
        slm = TheanoSLM(
                in_shape=(batchsize,) + X.shape[1:] + (1,),
                description=desc)
    elif X.ndim == 4:
        slm = TheanoSLM(
                in_shape=(batchsize,) + X.shape[1:],
                description=desc)
    else:
        raise NotImplementedError()
    slm.get_theano_fn()  # -- pre-compile the feature extractor

    extractor = FeatureExtractor(X, slm,
            filename=filename,
            batchsize=batchsize,
            TEST=False)
    extractor.compute_features(use_memmap=True)


def main_view2_features_from_seed():
    _, cmd, filename, seed = sys.argv

    config = Bandit1().template.sample(int(seed))
    return _main_featureextract_helper(config)


def main_view2_features_from_danpkl():
    _, _cmd, rank, pklfile = sys.argv

    obj = cPickle.load(open(pklfile))
    configs = [o['spec'] for o in obj]
    results = [o['result'] for o in obj]
    losses = [r['loss'] for r in results]
    # -- losses are sorted
    return _main_featureextract_helper(configs[int(rank)],
            'features_danpkl_%i.npy' % int(rank),
            splits=['fold_%i' % ii for ii in xrange(10)])

def main_view1_features_from_danpkl():
    _, _cmd, rank, pklfile = sys.argv

    obj = cPickle.load(open(pklfile))
    configs = [o['spec'] for o in obj]
    results = [o['result'] for o in obj]
    losses = [r['loss'] for r in results]
    # -- losses are sorted
    return _main_featureextract_helper(configs[int(rank)],
            'view1_features_danpkl_%i.npy' % int(rank),
            splits=['DevTrain', 'DevTest'])


def main_view1_classify():
    # usage: comparisons outprefix trace_normalize in_filename0 in_filename1 ...
    comparisons = [getattr(comp_module, comp)
            for comp in sys.argv[2].split(',')]
    out_prefix = sys.argv[3]
    trace_normalize = int(sys.argv[4])
    in_filenames = sys.argv[5:]

    dataset = skdata.lfw.Aligned()

    # n_out_features: after blending and whatnot, we'll have this many
    # features per example for the classifier.
    n_out_features = np.sum(
            [comparison.get_num_features(
                cPickle.load(open(filename + '.info'))[1])
                for filename in in_filenames
                for comparison in comparisons])
    out_features_pos = 0

    train_memmap = None
    test_memmap = None
    valid_memmap = None

    # this script is going to create three memmaps: train_X, test_X, valid_X
    # each memmap will use the first segments of columns for feature 1, the
    # second segment for feature 2 and so on.

    Xs = []
    train_y, test_y, valid_y = None, None, None
    for in_filename in in_filenames:
        in_dtype, in_shape = cPickle.load(open(in_filename + '.info'))
        features = np.memmap(in_filename, dtype=in_dtype, mode='r', shape=in_shape)
        print('loaded features of shape %s' % str(features.shape))

        for comparison in comparisons:
            pf = PairFeatures(dataset, features,
                    comparison=comparison,
                    filename_prefix='pairs')
            # -- override the default mapping because our feature file
            #    generated by main_view2_classify only has rows from the test
            #    folds
            pf.idx_of_path = {}
            __X, __y, Xpaths = get_relevant_images(dataset,
                    splits=['DevTrain', 'DevTest'],
                    dtype='float32',
                    strip_path=False)
            for ii, pth in enumerate(Xpaths):
                pf.idx_of_path[str(pth)] = ii

            train_Xy = pf.match_task(['DevTrain'])
            test_Xy = pf.match_task(['DevTest'])

            train_X, train_y_i = train_Xy
            test_X, test_y_i = test_Xy

            if train_y is None:
                train_y = train_y_i
                test_y = test_y_i
            else:
                assert np.all(train_y == train_y_i)
                assert np.all(test_y == test_y_i)

            train_mean, train_std = mean_and_std(train_X, min_std=1e-8)
            # -- each X is an in-memory copy of elements from a read-only mem-map
            for X in [train_X, test_X]:
                X -= train_mean  # a row vector
                X /= train_std   # a row vector
            if trace_normalize:
                print "SKIPPING TRACE NORMALIZATION"
            else:
                # -- "trace normalization" for blending features of different sizes
                train_l2 = np.mean(np.sqrt((train_X ** 2).sum(axis=1)))
                for X in [train_X, test_X]:
                    X /= train_l2

            # -- do not shuffle the training set for better SGD
            #    because it is in a memmap, too slow, not worth it

            # -- write train_X, test_X to a swath of the memmap
            if train_memmap is None:
                # now we know how big to make these files
                train_memmap = np.memmap(out_prefix + '.train',
                    dtype='float32',
                    mode='w+',
                    shape=(len(train_X), n_out_features))

                test_memmap = np.memmap(out_prefix + '.test',
                    dtype='float32',
                    mode='w+',
                    shape=(len(test_X), n_out_features))

            L = out_features_pos
            train_memmap[:, L:L + train_X.shape[1]] = train_X
            test_memmap [:, L:L + test_X.shape[1]] = test_X
            out_features_pos += train_X.shape[1]

    assert out_features_pos == train_memmap.shape[1]

    model, earlystopper, data = train_classifier(
            (train_memmap, train_y),
            (test_memmap, test_y),
            verbose=True,
            step_sizes=(1e-1, 3e-2, 1e-2, 3e-3, 1e-3),
            )
    print 'best y', earlystopper.best_y
    print 'best time', earlystopper.best_time

def main_view2_classify():
    # usage: comparison split outprefix trace_normalize in_filename0 in_filename1 ...
    comparisons = [getattr(comp_module, comp)
            for comp in sys.argv[2].split(',')]
    view2_split = int(sys.argv[3])
    out_prefix = sys.argv[4]
    trace_normalize = int(sys.argv[5])
    in_filenames = sys.argv[6:]

    dataset = skdata.lfw.Aligned()

    # n_out_features: after blending and whatnot, we'll have this many
    # features per example for the classifier.
    n_out_features = np.sum(
            [comparison.get_num_features(
                cPickle.load(open(filename + '.info'))[1])
                for filename in in_filenames
                for comparison in comparisons])
    out_features_pos = 0

    train_memmap = None
    test_memmap = None
    valid_memmap = None

    # this script is going to create three memmaps: train_X, test_X, valid_X
    # each memmap will use the first segments of columns for feature 1, the
    # second segment for feature 2 and so on.

    Xs = []
    train_y, test_y, valid_y = None, None, None
    for in_filename in in_filenames:
        in_dtype, in_shape = cPickle.load(open(in_filename + '.info'))
        features = np.memmap(in_filename, dtype=in_dtype, mode='r', shape=in_shape)
        print('loaded features of shape %s' % str(features.shape))

        for comparison in comparisons:
            pf = PairFeatures(dataset, features,
                    comparison=comparison,
                    filename_prefix='pairs')
            # -- override the default mapping because our feature file
            #    generated by main_view2_classify only has rows from the test
            #    folds
            pf.idx_of_path = {}
            __X, __y, Xpaths = get_relevant_images(dataset,
                    splits=['fold_%i' % ii for ii in xrange(10)],
                    dtype='float32',
                    strip_path=False)
            for ii, pth in enumerate(Xpaths):
                pf.idx_of_path[str(pth)] = ii

            train_splits = range(10)
            train_splits.remove(view2_split)
            train_splits.remove((view2_split + 1) % 10)

            train_Xy = pf.match_task( ['fold_%i' % ii for ii in train_splits])
            valid_Xy = pf.match_task('fold_%i' % ((view2_split + 1) % 10))
            test_Xy = pf.match_task('fold_%i' % view2_split)

            train_X, train_y_i = train_Xy
            test_X, test_y_i = test_Xy
            valid_X, valid_y_i = valid_Xy

            if train_y is None:
                train_y = train_y_i
                test_y = test_y_i
                valid_y = valid_y_i
            else:
                assert np.all(train_y == train_y_i)
                assert np.all(test_y == test_y_i)
                assert np.all(valid_y == valid_y_i)

            train_mean, train_std = mean_and_std(train_X, min_std=1e-8)
            # -- each X is an in-memory copy of elements from a read-only mem-map
            for X in [train_X, test_X, valid_X]:
                X -= train_mean  # a row vector
                X /= train_std   # a row vector
            # -- "trace normalization" for blending features of different sizes
            if trace_normalize:
                train_l2 = np.mean(np.sqrt((train_X ** 2).sum(axis=1)))
                for X in [train_X, test_X, valid_X]:
                    X /= train_l2

            # -- do not shuffle the training set for better SGD
            #    because it is in a memmap, too slow, not worth it

            # -- write train_X, test_X, valid_X to a swath of the memmap
            if train_memmap is None:
                # now we know how big to make these files
                train_memmap = np.memmap(out_prefix + '.train',
                    dtype='float32',
                    mode='w+',
                    shape=(len(train_X), n_out_features))

                test_memmap = np.memmap(out_prefix + '.test',
                    dtype='float32',
                    mode='w+',
                    shape=(len(test_X), n_out_features))

                valid_memmap = np.memmap(out_prefix + '.valid',
                    dtype='float32',
                    mode='w+',
                    shape=(len(valid_X), n_out_features))

            L = out_features_pos
            train_memmap[:, L:L + train_X.shape[1]] = train_X
            valid_memmap[:, L:L + valid_X.shape[1]] = valid_X
            test_memmap [:, L:L + test_X.shape[1]] = test_X
            out_features_pos += train_X.shape[1]

    assert out_features_pos == train_memmap.shape[1]

    model, earlystopper, data = train_classifier(
            (train_memmap, train_y),
            (valid_memmap, valid_y),
            verbose=True,
            step_sizes=(1e-1, 3e-2, 1e-2, 3e-3, 1e-3))
    print 'best y', earlystopper.best_y
    print 'best time', earlystopper.best_time

    pred_y = model.predict(test_memmap)
    if pred_y.min() < 0:
        pred_y = (pred_y + 1) / 2
    if test_y.min() < 0:
        test_y = (test_y + 1) / 2
    assert set(pred_y) == set(test_y)  # catch a 0/1 vs -1/+1 error
    test_err = (pred_y != test_y).astype('float').mean()
    print 'test error', test_err


def main_splits_overlap():
    dataset = skdata.lfw.Aligned()
    lpaths0, rpaths0, labels0 = dataset.raw_verification_task(
            split='DevTrain')
    lpaths1, rpaths1, labels1 = dataset.raw_verification_task(
            split='DevTest')

    set0 = set(lpaths0)
    set0.update(rpaths0)

    set1 = set(lpaths1)
    set1.update(rpaths1)

    print len(set0)
    print len(set1)
    print len(set0.intersection(set1))
    print len(set0.union(set1))


if __name__ == '__main__':
    cmd = sys.argv[1]
    main = globals()['main_' + cmd]
    sys.exit(main())