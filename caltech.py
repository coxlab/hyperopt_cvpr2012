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

from theano_slm import (TheanosSLM, 
                        ExtractedFeatures, 
                        use_memmap)

        
DEFAULT_COMPARISONS = ['mult', 'absdiff', 'sqrtabsdiff', 'sqdiff']


class CaltechBandit(gb.GensonBandit):
    source_string = cvpr_params.string(cvpr_params.config)
    
    def __init__(self):
        super(CaltechBandit, self).__init__(source_string=self.source_string)

    @classmethod
    def evaluate(cls, config, ctrl, use_theano=True):
        result = get_performance(None, config, use_theano)
        return result


class CaltechBanditHetero(CaltechBandit):
    source_string = cvpr_params.string(cvpr_params.config_h)
          

def get_config_string(configs):
    return hashlib.sha1(repr(configs)).hexdigest()


def get_performance(outfile, config, use_theano=True):
    import skdata.caltech

    dataset = skdata.caltech.Caltech101()

    c_hash = get_config_string(config)

    num_splits = 10
    batchsize = 4
    
    all_arrays, all_names = get_relevant_images(dataset, num_splits)
     
    performance_comp = {}
    feature_file_name = 'features_' + c_hash + '.dat'


    with TheanoExtractedFeatures(all_arrays, batchsize, config['desc'], 
                                     feature_file_name) as feature_fp:
        perfs = []
        for split_id in range(num_splits):
             
            train_names, ytrain = dataset.raw_classification_task(split='train_' + str(split_id))
            test_names, ytest = dataset.raw_classification_task(split='test_' + str(split_id))

            train_inds = np.searchsorted(all_names, train_names)
			test_inds =  np.searchsorted(all_names, test_names)
			
			train_features = feature_fp[train_inds]
			train_Xy = (train_features, ytrain)
			
			test_features = features_fp[test_inds]
			test_Xy = (test_features, ytest)
			
			perf = train_classifier(config, train_Xy, test_Xy, n_features)
            perfs.append(perf)
            
    performance = float(np.array(perfs).mean())
    result = dict(loss=performance, status='ok')
    
    if outfile is not None:
        outfh = open(outfile,'w')
        cPickle.dump(result, outfh)
        outfh.close()
    return result


class SplitFeatures(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def work(self, dataset, split, X, n_features,
             feature_fp, filename):
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


def get_relevant_images(dataset, dtype='uint8'):

    Xr, yr = dataset.raw_classification_task()
    Xr = np.array(Xr)

    Atr, Btr, c = dataset.raw_verification_task_resplit(split='train_0')
    Ate, Bte, c = dataset.raw_verification_task_resplit(split='test_0')
    all_images = np.unique(np.concatenate([Atr,Btr,Ate,Bte]))

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