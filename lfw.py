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

import theano
import theano.tensor as tensor
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from pythor3.model import SequentialLayeredModel

import skdata.larray
import skdata.utils
import hyperopt.genson_bandits as gb

try:
    import sge_utils
except ImportError:
    pass
import cvpr_params
import comparisons as comp_module
from theano_slm import (TheanoExtractedFeatures, 
                        train_classifier, 
                        use_memmap)

        
DEFAULT_COMPARISONS = ['mult', 'absdiff', 'sqrtabsdiff', 'sqdiff']


class LFWBandit(gb.GensonBandit):
    source_string = cvpr_params.string(cvpr_params.config)
    
    def __init__(self):
        super(LFWBandit, self).__init__(source_string=self.source_string)

    @classmethod
    def evaluate(cls, config, ctrl, use_theano=True):
        result = get_performance(None, config, use_theano)
        return result


class LFWBanditHetero(LFWBandit):
    source_string = cvpr_params.string(cvpr_params.config_h)
      

class LFWBanditHetero2(LFWBandit):
    source_string = cvpr_params.string(cvpr_params.config_h2) 
    

class LFWBanditSGE(LFWBandit):
    @classmethod
    def evaluate(cls, config, ctrl, use_theano=True):
        outfile = os.path.join('/tmp',get_config_string(config))
        opstring = '-l qname=hyperopt.q -o /home/render/hyperopt_jobs -e /home/render/hyperopt_jobs'
        jobid = sge_utils.qsub(get_performance, (outfile, config, use_theano),
                     opstring=opstring)
        status = sge_utils.wait_and_get_statuses([jobid])
        return cPickle.loads(open(outfile).read())
    

def get_config_string(configs):
    return hashlib.sha1(repr(configs)).hexdigest()


def get_performance(outfile, config, use_theano=True):
    import skdata.lfw
    dataset = skdata.lfw.Aligned()

    c_hash = get_config_string(config)

    comparisons = config.get('comparisons',DEFAULT_COMPARISONS)    
    assert all([hasattr(comp_module,comparison) for comparison in comparisons])

    num_splits = 1
    
    X, y, Xr = get_relevant_images(dataset, num_splits, dtype='float32')

    batchsize = 4
    
    performance_comp = {}
    feature_file_name = 'features_' + c_hash + '.dat'
    train_pairs_filename = 'train_pairs_' + c_hash + '.dat'
    test_pairs_filename = 'test_pairs_' + c_hash + '.dat' 
    with TheanoExtractedFeatures(X, batchsize, config['desc'], 
                                       feature_file_name) as features_fp:
                                       
        feature_shp = features_fp.feature_shp
        for comparison in comparisons:
            print('Doing comparison %s' % comparison)
            perf = []
            comparison_obj = getattr(comp_module,comparison)
            n_features = comparison_obj.get_num_features(feature_shp)
            for split_id in range(num_splits):
                with PairFeatures(dataset, 'train_' + str(split_id), Xr,
                        n_features, features_fp, comparison_obj,
                                  train_pairs_filename) as train_Xy:
                    with PairFeatures(dataset, 'test_' + str(split_id),
                            Xr, n_features, features_fp, comparison_obj,
                                      test_pairs_filename) as test_Xy:
                        perf.append(train_classifier(config,
                                    train_Xy, test_Xy, n_features))
            performance_comp[comparison] = float(np.array(perf).mean())
            
    performance = float(np.array(performance_comp.values()).min())
    result = dict(loss=performance, performances=performance_comp, status='ok')
    
    if outfile is not None:
        outfh = open(outfile,'w')
        cPickle.dump(result, outfh)
        outfh.close()
    return result


class PairFeatures(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def work(self, dataset, split, X, n_features,
             feature_fp, comparison_obj, filename):
        A, B, labels = dataset.raw_verification_task_resplit(split=split)
        Ar = np.array([os.path.split(ar)[-1] for ar in A])
        Br = np.array([os.path.split(br)[-1] for br in B])
        Aind = np.searchsorted(X, Ar)
        Bind = np.searchsorted(X, Br)
        assert len(Aind) == len(Bind)
        pair_shp = (len(labels), n_features)
        
        size = 4 * np.prod(pair_shp)
        print('Total size: %i bytes (%.2f GB)' % (size, size / float(1e9)))                            
        memmap = use_memmap(size)       
        if memmap:
            print('get_pair_fp memmap %s for features of shape %s' % (
                                                    filename, str(pair_shp)))
            feature_pairs_fp = np.memmap(filename,
                                    dtype='float32',
                                    mode='w+',
                                    shape=pair_shp)
        else:
            print('using memory for features of shape %s' % str(pair_shp))
            feature_pairs_fp = np.empty(pair_shp, dtype='float32')                                    

        for (ind,(ai, bi)) in enumerate(zip(Aind, Bind)):
            feature_pairs_fp[ind] = comparison_obj(feature_fp[ai],
                                                   feature_fp[bi])
            if ind % 100 == 0:
                print('get_pair_fp  %i / %i' % (ind, len(Aind)))

        if memmap:                
            print ('flushing memmap')
            sys.stdout.flush()
            del feature_pairs_fp
            self.filename = filename
            self.features = np.memmap(filename,
                    dtype='float32',
                    mode='r',
                    shape=pair_shp)
        else:
            self.features = feature_pairs_fp
            self.filename = ''
            
        self.labels = labels
        
    def __enter__(self):
        self.work(*self.args, **self.kwargs)
        return (self.features, self.labels)

    def __exit__(self, *args):
        if self.filename:
            os.remove(self.filename)



######utils

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


def get_relevant_images(dataset, num_splits, dtype='uint8'):
    # load & resize logic is LFW Aligned -specific
    assert 'Aligned' in str(dataset.__class__)

    Xr, yr = dataset.raw_classification_task()
    Xr = np.array(Xr)

    dsets = []
    for ind in range(num_splits):
        Atr, Btr, c = dataset.raw_verification_task_resplit(split='train_' + str(ind))
        Ate, Bte, c = dataset.raw_verification_task_resplit(split='test_' + str(ind))
        dsets.extend([Atr, Btr, Ate, Bte])
    all_images = np.unique(np.concatenate(dsets))

    inds = np.searchsorted(Xr, all_images)
    Xr = Xr[inds]
    yr = yr[inds]

    X = skdata.larray.lmap(
                ImgLoaderResizer(
                    shape=(200, 200),  # lfw-specific
                    dtype=dtype),
                Xr)

    Xr = np.array([os.path.split(x)[-1] for x in Xr])

    return X, yr, Xr