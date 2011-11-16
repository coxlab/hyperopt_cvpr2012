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

from theano_slm import TheanoExtractedFeatures, train_multiclassifier

        
class CaltechBandit(gb.GensonBandit):
    source_string = cvpr_params.string(cvpr_params.config)
    
    def __init__(self):
        super(CaltechBandit, self).__init__(source_string=self.source_string)

    @classmethod
    def evaluate(cls, config, ctrl, use_theano=True):
        result = get_performance(None, config, cls.dset_name, use_theano=use_theano)
        return result


class Caltech101Bandit(CaltechBandit):
    dset_name = 'Caltech101'

class Caltech256Bandit(CaltechBandit):
    dset_name = 'Caltech256'


def get_config_string(configs):
    return hashlib.sha1(repr(configs)).hexdigest()


def get_performance(outfile, config, dset_string, use_theano=True):
    import skdata.caltech

    dataset = getattr(skdata.caltech, dset_string)()

    c_hash = get_config_string(config)

    num_splits = 10
    batchsize = 4
    
    arrays, paths = get_relevant_images(dataset, num_splits)
     
    performance_comp = {}
    feature_file_name = 'features_' + c_hash + '.dat'

    with TheanoExtractedFeatures(all_arrays, batchsize, [config], 
                                     [feature_file_name]) as features_fps:

        features_fp = features_fps[0]
        perfs = []
        for split_id in range(num_splits):             
            train_names, ytrain = dataset.raw_classification_task(split='train_' + str(split_id))
            train_names = get_paths(train_names)
            test_names, ytest = dataset.raw_classification_task(split='test_' + str(split_id))
            test_names = get_paths(test_names)

            train_inds = np.searchsorted(paths, train_names)
			test_inds =  np.searchsorted(all_names, test_names)
			
			train_features = features_fp[train_inds]
			train_Xy = (train_features, ytrain)
			
			test_features = features_fp[test_inds]
			test_Xy = (test_features, ytest)
			
			perf = train_multiclassifier(config, train_Xy, test_Xy, n_features)
            perfs.append(perf)
            
    performance = float(np.array(perfs).mean())
    result = dict(loss=performance, status='ok')
    
    if outfile is not None:
        outfh = open(outfile,'w')
        cPickle.dump(result, outfh)
        outfh.close()
    return result


class ImgLoaderResizer(object):
    """ Load variously-sized rgb images, return normalized 200x200 float32 ones.
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


def get_paths(X):
    return np.array([os.path.split(x)[-1] for x in X])
    

def get_relevant_images(dataset, num_splits, dtype='uint8'):

    X, yr = dataset.raw_classification_task()
    Xr = get_paths(X)

    dsets = []
    for ind in range(num_splits):
        Atr, b = dataset.raw_classification_task(split='train_' + str(ind))
        Ate, b = dataset.raw_classification_task(split='test_' + str(ind))
        dsets.extend([Atr, Ate])
    all_images = np.unique(np.concatenat(dsets))

    inds = np.searchsorted(Xr, all_images)
    Xr = Xr[inds]
    yr = yr[inds]

    arrays = skdata.larray.lmap(
                ImgLoaderResizer(
                    shape=(200, 200),  
                    dtype=dtype),
                X)
    
    return arrays, Xr