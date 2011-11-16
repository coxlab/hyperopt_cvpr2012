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
import hyperopt.genson_bandits as gb
from hyperopt.mongoexp import MongoJobs, MongoExperiment, as_mongo_str

# importing here somehow makes unpickling work, despite circular import in
# hyperopt
import hyperopt.theano_bandit_algos

from utils import TooLongException
from theano_slm import TheanoSLM, interpret_model
from classifier import train_classifier, split_center_normalize


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
        result = get_performance(None, son_to_py(config), use_theano)
        return result


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


class FeatureExtractor(object):
    def __init__(self, X, slm,
            tlimit=float('inf'),
            batchsize=4,
            filename='FeatureExtractor.npy',
            TEST=False):
        """
        X - 4-tensor of images
        feature_shp - 4-tensor of output feature shape (len matches X)
        batchsize - number of features to extract in parallel
        slm - feature-extraction module (with .process_batch() fn)
        filename - store features to memmap here

        returns - read-only memmap of features
        """
        self.filename = filename
        self.batchsize = batchsize
        self.tlimit = tlimit
        self.X = X
        self.slm = slm
        self.verbose = False
        self.n_to_extract = len(X)
        if TEST:
            print('FeatureExtractor running in TESTING mode')
            self.verbose = True
            self.n_to_extract = 10 * batchsize
        assert self.n_to_extract <= len(X)

        # -- convenience
        self.feature_shp = (self.n_to_extract,) + self.slm.pythor_out_shape

    def __enter__(self):
        raise NotImplementedError()
        #return self.features

    def __exit__(self, *args):
        raise NotImplementedError()
        for filename in self.filenames:
            if filename:
                os.remove(filename)

    def extract_to_memmap(self):
        """
        Allocate a memmap, fill it with extracted features, return r/o view.
        """
        filename = self.filename
        feature_shp = self.feature_shp
        print('Creating memmap %s for features of shape %s' % (
                                              filename,
                                              str(feature_shp)))
        features_fp = np.memmap(filename,
            dtype='float32',
            mode='w+',
            shape=feature_shp)
        info = open(filename+'.info', 'w')
        cPickle.dump(('float32', feature_shp), info)
        del info

        self.extract_to_storage(features_fp)

        # -- docs here:
        #    http://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html
        #    say that deletion is the way to flush changes !?
        del features_fp
        rval = np.memmap(self.filename,
            dtype='float32',
            mode='r',
            shape=feature_shp)
        return rval

    def extract_to_storage(self, arr):
        """
        Fill arr with the first len(arr) features of self.X.
        """
        assert len(arr) <= len(self.X)
        batchsize = self.batchsize
        tlimit = self.tlimit
        print('Total size: %i bytes (%.2f GB)' % (
            arr.size * arr.dtype.itemsize,
            arr.size * arr.dtype.itemsize / float(1e9)))
        i = 0
        t0 = time.time()
        while True:
            if i + batchsize >= len(arr):
                assert i < len(arr)
                xi = np.asarray(self.X[-batchsize:])
                done = True
            else:
                xi = np.asarray(self.X[i:i+batchsize])
                done = False
            t1 = time.time()
            feature_batch = self.slm.process_batch(xi)
            if self.verbose:
                print('compute: ', time.time() - t1)
            t2 = time.time()
            delta = max(0, i + batchsize - len(arr))
            assert np.all(np.isfinite(feature_batch))
            arr[i:i + batchsize - delta] = feature_batch[delta:]
            if self.verbose:
                print('write: ', time.time() - t2)
            if done:
                break

            i += batchsize
            if (i // batchsize) % 50 == 0:
                t_cur = time.time() - t0
                t_per_image = (time.time() - t0) / i
                t_tot = t_per_image * len(arr)
                if tlimit is not None and t_tot / 60.0 > tlimit:
                    raise TooLongException(t_tot/60.0, tlimit)
                print 'extraction: %i / %i  mins: %.2f / %.2f ' % (
                        i , len(arr),
                        t_cur / 60.0, t_tot / 60.0)

    def compute_features(self, use_memmap=None):
        if use_memmap is None:
            size = np.prod(self.feature_shp) * 4
            use_memmap = (size > 3e8)  # 300MB cutoff

        if use_memmap:
            return self.extract_to_memmap()
        else:
            print('Using memory for features of shape %s' % str(self.feature_shp))
            arr = np.empty(self.feature_shp, dtype='float32')
            self.extract_to_storage(arr)
            return arr


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

    def view1_train_match_task(self):
        """
        Returns an image verification task
        """
        lpaths, rpaths, labels = self.dataset.raw_verification_task(
                split='DevTrain')
        arr = self.compute_pairs_features(lpaths, rpaths)
        return arr, labels

    def view1_test_match_task(self):
        lpaths, rpaths, labels = self.dataset.raw_verification_task(
                split='DevTest')
        arr = self.compute_pairs_features(lpaths, rpaths)
        return arr, labels

    def view1_resplit(self, name):
        lpaths, rpaths, labels = self.dataset.raw_verification_task_resplit(
                split=name)
        arr = self.compute_pairs_features(lpaths, rpaths)
        return arr, labels

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

    def view1_train_match_task_fliplr(self):
        """
        Returns an image verification task
        """
        lpaths, rpaths, labels = self.dataset.raw_verification_task(
                split='DevTrain')
        return self.compute_pairs_features_fliplr(lpaths, rpaths, labels)

    def view1_resplit_fliplr(self, name):
        lpaths, rpaths, labels = self.dataset.raw_verification_task_resplit(
                split=name)
        return self.compute_pairs_features_fliplr(lpaths, rpaths, labels)

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
    _, cmd, in_filename, comparison, flip_lr = sys.argv

    in_dtype, in_shape = cPickle.load(open(in_filename + '.info'))
    features = np.memmap(in_filename, dtype=in_dtype, mode='r', shape=in_shape)
    print('loaded features of shape %s' % str(features.shape))
    dataset = skdata.lfw.Aligned()

    flip_lr = bool(int(flip_lr))

    pf = PairFeatures(dataset, features,
            comparison=getattr(comp_module, comparison),
            filename_prefix='pairs')

    if 1:
        print "OPTIMIZING TEST SET PERFORMANCE OF OFFICIAL DEV SPLIT"
        if flip_lr:
            train_Xy = pf.view1_train_match_task_fliplr()
        else:
            train_Xy = pf.view1_train_match_task()
        test_Xy = pf.view1_test_match_task()

        train_X, train_y = train_Xy
        test_X, test_y = test_Xy

        m = np.mean(train_X, axis=0)
        s = np.std(train_X, axis=0)
        train_X = (train_X - m) / s
        test_X = (test_X - m) / s

        model, earlystopper = train_classifier(
                (train_X, train_y),
                (test_X, test_y), verbose=True)
        print 'best y', earlystopper.best_y
        print 'best time', earlystopper.best_time
    else:
        print "OPTIMIZING TEST SET PERFORMANCE OF RESPLIT 0"
        if flip_lr:
            train_Xy = pf.view1_resplit_fliplr('train_0')
        else:
            train_Xy = pf.view1_resplit('train_0')

        test_Xy = pf.view1_resplit('test_0')
        train_X, train_y = train_Xy
        test_X, test_y = test_Xy
        m = np.mean(train_X, axis=0)
        s = np.std(train_X, axis=0)
        train_X = (train_X - m) / s
        test_X = (test_X - m) / s
        model, earlystopper = train_classifier(
                (train_X, train_y),
                (test_X, test_y),
                verbose=True,
                #step_sizes=[1e-6],
                )
        print 'best y', earlystopper.best_y
        print 'best time', earlystopper.best_time


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


if 0:

    def get_performance(outfile, config, use_theano=True):
        import skdata.lfw

        c_hash = get_config_string(config)

        comparisons = config['comparisons']

        assert all([hasattr(comp_module, comparison)
            for comparison in comparisons])

        dataset = skdata.lfw.Aligned()

        X, y, Xr = get_relevant_images(dataset, dtype='float32')

        assert len(X) == len(dataset.meta)

        batchsize = 4

        desc = copy.deepcopy(config['desc'])
        interpret_model(desc)
        if X.ndim == 3:
            theano_slm = TheanoSLM(
                    in_shape=(batchsize,) + X.shape[1:] + (1,),
                    description=desc)
        elif X.ndim == 4:
            theano_slm = TheanoSLM(
                    in_shape=(batchsize,) + X.shape[1:],
                    description=desc)
        else:
            raise NotImplementedError()

        if use_theano:
            slm = theano_slm
            # -- pre-compile function to not mess up timing
            slm.get_theano_fn()
        else:
            cthor_sse = {'plugin':'cthor', 'plugin_kwargs':{'variant':'sse'}}
            cthor = {'plugin':'cthor', 'plugin_kwargs':{}}
            slm = SequentialLayeredModel(X.shape[1:], desc,
                                         plugin='passthrough',
                                         plugin_kwargs={'plugin_mapping': {
                                             'fbcorr': cthor,
                                              'lnorm' : cthor,
                                              'lpool' : cthor,
                                         }})

        feature_shp = (X.shape[0],) + theano_slm.pythor_out_shape

        num_splits = 1
        performance_comp = {}
        feature_file_name = 'features_' + c_hash + '.dat'
        train_pairs_filename = 'train_pairs_' + c_hash + '.dat'
        test_pairs_filename = 'test_pairs_' + c_hash + '.dat'
        with FeatureExtractor(X, feature_shp, batchsize, slm,
                feature_file_name) as features_mmap:
            assert len(features_mmap) == len(X)
            for comparison in comparisons:
                print('Doing comparison %s' % comparison)
                perf = []
                comparison_obj = getattr(comp_module,comparison)
                n_features = comparison_obj.get_num_features(feature_shp)
                for split_id in range(num_splits):
                    with PairFeatures(
                            dataset,
                            'train_' + str(split_id),
                            Xr,
                            n_features,
                            features_mmap,
                            comparison_obj,
                            train_pairs_filename) as train_Xy:
                        with PairFeatures(
                                dataset,
                                'test_' + str(split_id),
                                Xr,
                                n_features,
                                features_mmap,
                                comparison_obj,
                                test_pairs_filename) as test_Xy:
                            #
                            perf.append(train_classifier(config,
                                        train_Xy, test_Xy, n_features))
                            n_test_examples = len(test_Xy[0])
                performance_comp[comparison] = float(np.array(perf).mean())

        performance = float(np.array(performance_comp.values()).min())
        result = dict(
                loss=performance,
                loss_variance=performance * (1 - performance) / n_test_examples,
                performances=performance_comp,
                status='ok')

        if outfile is not None:
            outfh = open(outfile,'w')
            cPickle.dump(result, outfh)
            outfh.close()
        return result


    def get_relevant_images(dataset, dtype):
        """Create a lazy-array of images that matches dataset.meta

        Returns lazyarray of images, array of ints for names, array of img
        filenames
        """
        # load & resize logic is LFW Aligned -specific
        assert 'Aligned' in str(dataset.__class__)

        Xr, yr = dataset.raw_classification_task()
        Xr = np.array(Xr)
        # Xr is image filenames
        # yr is ints corresponding to names

        X = skdata.larray.lmap(
                    ImgLoaderResizer(
                        shape=(200, 200),  # lfw-specific
                        dtype=dtype),
                    Xr)

        Xr = np.array([os.path.split(x)[-1] for x in Xr])

        return X, yr, Xr


