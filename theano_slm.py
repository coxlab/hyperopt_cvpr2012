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
from pythor3.model.slm.plugins.passthrough.passthrough import (
        SequentialLayeredModelPassthrough,
        )
from pythor3.model import SequentialLayeredModel
from pythor3.operation import fbcorr_
from pythor3.operation import lnorm_

import asgd  # use master branch from https://github.com/jaberg/asgd
import skdata.larray
import skdata.utils
import hyperopt.genson_bandits as gb

try:
    import sge_utils
except ImportError:
    pass
import cvpr_params
from early_stopping import fit_w_early_stopping, EarlyStopping



class TheanoSLM(object):
    """
    SequentialLayeredModel clone implemented with Theano
    """

    def __init__(self, in_shape, description,
            dtype='float32', rng=888):

        # -- transpose shape to theano-convention (channel major)

        if len(in_shape) == 2:
            self.theano_in_shape = (1, 1,) +  in_shape
            self.pythor_in_shape = in_shape
        elif len(in_shape) == 3:
            self.theano_in_shape = (1, in_shape[2], in_shape[0], in_shape[1])
            self.pythor_in_shape = in_shape
        else:
            self.theano_in_shape = (in_shape[0],
                    in_shape[3],
                    in_shape[1],
                    in_shape[2])
            self.pythor_in_shape = in_shape[1:]

        assert len(self.theano_in_shape) == 4
        print 'TheanoSLM.theano_in_shape', self.theano_in_shape

        # This guy is used to generate filterbanks
        pythor_safe_description = get_pythor_safe_description(description)
        try:
            self.SLMP = SequentialLayeredModelPassthrough(
                    self.pythor_in_shape,
                    pythor_safe_description,
                    dtype=dtype)
        except ValueError, e:
            if 'negative dimensions' in str(e):
                print 'pythor_in_shape', self.pythor_in_shape
                print 'in_shape', in_shape
                raise InvalidDescription()
            raise

        del in_shape

        self.s_input = tensor.ftensor4('arr_in')
        self.rng = np.random.RandomState(rng)  # XXX check for rng being int

        x = self.s_input
        x_shp = self.theano_in_shape
        for layer_idx, layer_desc in enumerate(description):
            for op_name, op_params in layer_desc:
                init_fn = getattr(self, 'init_' + op_name)
                _D = dict_add(
                            op_params.get('kwargs', {}),
                            op_params.get('initialize', {}))
                x, x_shp = init_fn(x, x_shp, **_D)
                print 'added layer', op_name, 'shape', x_shp

        if 0 == np.prod(x_shp):
            raise InvalidDescription()

        self.theano_out_shape = x_shp
        self.pythor_out_shape = x_shp[2], x_shp[3], x_shp[1]
        self.s_output = x

    def init_fbcorr_h(self, x, x_shp, **kwargs):      
        min_out = kwargs.get('min_out', fbcorr_.DEFAULT_MIN_OUT)
        max_out = kwargs.get('max_out', fbcorr_.DEFAULT_MAX_OUT)
        kwargs['max_out'] = get_into_shape(max_out)
        kwargs['min_out'] = get_into_shape(min_out)       
        return self.init_fbcorr(x, x_shp, **kwargs)
            
    def init_fbcorr(self, x, x_shp, n_filters,
            filter_shape,
            min_out=fbcorr_.DEFAULT_MIN_OUT,
            max_out=fbcorr_.DEFAULT_MAX_OUT,
            stride=fbcorr_.DEFAULT_STRIDE,
            mode=fbcorr_.DEFAULT_MODE,
            generate=None):
        # Reference implementation:
        # ../pythor3/pythor3/operation/fbcorr_/plugins/scipy_naive/scipy_naive.py
        if stride != fbcorr_.DEFAULT_STRIDE:
            raise NotImplementedError('stride is not used in reference impl.')
        fake_x = np.empty((x_shp[2], x_shp[3], x_shp[1]),
                x.dtype)
        kerns = self.SLMP._get_filterbank(fake_x,
                dict(n_filters=n_filters,
                    filter_shape=filter_shape,
                    generate=generate))
        kerns = kerns.transpose(0, 3, 1, 2).copy()[:,:,::-1,::-1]
        x = conv.conv2d(
                x,
                kerns,
                image_shape=x_shp,
                filter_shape=kerns.shape,
                border_mode=mode)
        if mode == 'valid':
            x_shp = (x_shp[0], n_filters,
                    x_shp[2] - filter_shape[0] + 1,
                    x_shp[3] - filter_shape[1] + 1)
        elif mode == 'full':
            x_shp = (x_shp[0], n_filters,
                    x_shp[2] + filter_shape[0] - 1,
                    x_shp[3] + filter_shape[1] - 1)
        else:
            raise NotImplementedError('fbcorr mode', mode)

        if min_out is None and max_out is None:
            return x, x_shp
        elif min_out is None:
            return tensor.minimum(x, max_out), x_shp
        elif max_out is None:
            return tensor.maximum(x, min_out), x_shp
        else:
            return tensor.clip(x, min_out, max_out), x_shp

    def boxconv(self, x, x_shp, kershp, channels=False):
        """
        channels: sum over channels (T/F)
        """
        kershp = tuple(kershp)
        if channels:
            rshp = (   x_shp[0],
                        1,
                        x_shp[2] - kershp[0] + 1,
                        x_shp[3] - kershp[1] + 1)
            kerns = np.ones((1, x_shp[1]) + kershp, dtype=x.dtype)
        else:
            rshp = (   x_shp[0],
                        x_shp[1],
                        x_shp[2] - kershp[0] + 1,
                        x_shp[3] - kershp[1] + 1)
            kerns = np.ones((1, 1) + kershp, dtype=x.dtype)
            x_shp = (x_shp[0]*x_shp[1], 1, x_shp[2], x_shp[3])
            x = x.reshape(x_shp)
        try:
            rval = tensor.reshape(
                    conv.conv2d(x,
                        kerns,
                        image_shape=x_shp,
                        filter_shape=kerns.shape,
                        border_mode='valid'),
                    rshp)
        except Exception, e:
            if "Bad size for the output shape" in str(e):
                raise InvalidDescription()
            else:
                raise
        return rval, rshp

    def init_lnorm_h(self, x, x_shp, **kwargs):
        threshold = kwargs.get('threshold', lnorm_.DEFAULT_THRESHOLD)
        stretch = kwargs.get('stretch', lnorm_.DEFAULT_STRETCH)
        kwargs['threshold'] = get_into_shape(threshold)
        kwargs['stretch'] = get_into_shape(stretch)
        return self.init_lnorm(x, x_shp, **kwargs)

    def init_lnorm(self, x, x_shp,
            inker_shape=lnorm_.DEFAULT_INKER_SHAPE,    # (3, 3)
            outker_shape=lnorm_.DEFAULT_OUTKER_SHAPE,  # (3, 3)
            remove_mean=lnorm_.DEFAULT_REMOVE_MEAN,    # False
            div_method=lnorm_.DEFAULT_DIV_METHOD,      # 'euclidean'
            threshold=lnorm_.DEFAULT_THRESHOLD,        # 0.
            stretch=lnorm_.DEFAULT_STRETCH,            # 1.
            mode=lnorm_.DEFAULT_MODE,                  # 'valid'
            ):
        # Reference implementation:
        # ../pythor3/pythor3/operation/lnorm_/plugins/scipy_naive/scipy_naive.py
        EPSILON = lnorm_.EPSILON
        if mode != 'valid':
            raise NotImplementedError('lnorm requires mode=valid', mode)

        if outker_shape == inker_shape:
            size = np.asarray(x_shp[1] * inker_shape[0] * inker_shape[1],
                    dtype=x.dtype)
            ssq, ssqshp = self.boxconv(x ** 2, x_shp, inker_shape,
                    channels=True)
            xs = inker_shape[0] // 2
            ys = inker_shape[1] // 2
            if div_method == 'euclidean':
                if remove_mean:
                    arr_sum, _shp = self.boxconv(x, x_shp, inker_shape,
                            channels=True)
                    arr_num = x[:, :, xs:-xs, ys:-ys] - arr_sum / size
                    arr_div = EPSILON + tensor.sqrt(
                            tensor.maximum(0,
                                ssq - (arr_sum ** 2) / size))
                else:
                    arr_num = x[:, :, xs:-xs, ys:-ys]
                    arr_div = EPSILON + tensor.sqrt(ssq)
            else:
                raise NotImplementedError('div_method', div_method)
        else:
            raise NotImplementedError('outker_shape != inker_shape',outker_shape, inker_shape)
        if stretch != 1:
            arr_num = arr_num * stretch
            arr_div = arr_div * stretch
        arr_div = tensor.switch(arr_div < (threshold + EPSILON), 1.0, arr_div)

        r = arr_num / arr_div
        r_shp = x_shp[0], x_shp[1], ssqshp[2], ssqshp[3]
        return r, r_shp

    def init_lpool_h(self, **kwargs):
        order = kwargs.get('order', 1)
        kwargs['order'] = get_into_shape(order)
        return init_lpool(self, **kwargs)

    def init_lpool(self, x, x_shp,
            ker_shape=(3, 3),
            order=1,
            stride=1,
            mode='valid'):
        #XXX: respect kwargs and do correct math

        if order == 1:
            r, r_shp = self.boxconv(x, x_shp, ker_shape)
        else:
            r, r_shp = self.boxconv(x ** order, x_shp, ker_shape)
            r = tensor.maximum(r, 0) ** (1.0 / order)
        if stride > 1:
            r = r[:, :, ::stride, ::stride]
            # intdiv is tricky... so just use numpy
            r_shp = np.empty(r_shp)[:, :, ::stride, ::stride].shape
        return r, r_shp

    def get_theano_fn(self):
        try:
            fn = self._fn
        except AttributeError:
            fn = self._fn = theano.function([self.s_input], self.s_output,
                allow_input_downcast=True)
        return fn

    def process_batch(self, arr_in):
        fn = self.get_theano_fn()
        if arr_in.ndim == 4:
            channel_major_in = arr_in.transpose(0, 3, 1, 2)
        elif arr_in.ndim == 3:
            channel_major_in = arr_in[:,:,:,None].transpose(0, 3, 1, 2)
        else:
            raise NotImplementedError()
        return fn(channel_major_in).transpose(0, 2, 3, 1)

    def process(self, arr_in):
        """Return something like SequentialLayeredModel would have
        """
        rval = self.process_batch(arr_in[None,None,:,:])[0]
        if rval.shape[2] == 1:
            # -- drop the colour channel for single-channel images
            return rval[:, :, 0]
        else:
            return rval


def train_classifier(config, train_Xy, test_Xy, n_features):
    print 'training classifier'
    train_X, train_y = train_Xy
    test_X, test_y = test_Xy
    train_mean = train_X.mean(axis=0)
    train_std = train_X.std(axis=0)

    # -- change labels to -1, +1
    assert set(train_y) == set([0, 1])
    train_y = train_y * 2 - 1
    test_y = test_y * 2 - 1

    def normalize(XX):
        return (XX - train_mean) / np.maximum(train_std, 1e-6)

    model, earlystopper = fit_w_early_stopping(
            model=asgd.naive_asgd.NaiveBinaryASGD(
                n_features=n_features,
                l2_regularization=0,
                sgd_step_size0=1e-3),
            es=EarlyStopping(warmup=20), # unit: validation intervals
            train_X=normalize(train_X),
            train_y=train_y,
            validation_X=normalize(test_X),
            validation_y=test_y,
            batchsize=10,                # unit: examples
            validation_interval=100)     # unit: batches
    return earlystopper.best_y


class LFWBandit(gb.GensonBandit):
    def __init__(self):
        source_string = repr(cvpr_params.config).replace("'",'"')
        super(LFWBandit, self).__init__(source_string=source_string)

    @classmethod
    def evaluate(cls, config, ctrl, use_theano=True):
        result = get_performance(None, config, use_theano)
        return result


class LFWBanditSGE(LFWBandit):
    @classmethod
    def evaluate(cls, config, ctrl, use_theano=True):
        outfile = os.path.join('/tmp',get_config_string(config))
        opstring = '-l qname=hyperopt.q -o /home/render/hyperopt_jobs -e /home/render/hyperopt_jobs'
        jobid = sge_utils.qsub(get_performance, (outfile, config, use_theano),
                     opstring=opstring)
        status = sge_utils.wait_and_get_statuses([jobid])
        return cPickle.loads(open(outfile).read())
        
        
def get_performance(outfile, config, use_theano=True):
    import skdata.lfw

    c_hash = get_config_string(config)

    comparison = get_comparison(config)

    dataset = skdata.lfw.Aligned()

    X, y, Xr = get_relevant_images(dataset)

    batchsize = 4

    desc = config['desc']
    interpret_model(desc)
    if X.ndim == 3:
        theano_slm = TheanoSLM(
                in_shape=(batchsize,) + X.shape[1:] + (1,),
                description=config['desc'])
    elif X.ndim == 4:
        theano_slm = TheanoSLM(
                in_shape=(batchsize,) + X.shape[1:],
                description=config['desc'])
    else:
        raise NotImplementedError()

    if use_theano:
        slm = theano_slm
        # -- pre-compile the function to not mess up timing
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
    performances = []
    feature_file_name = 'features_' + c_hash + '.dat'
    train_pairs_filename = 'train_pairs_' + c_hash + '.dat'
    test_pairs_filename = 'test_pairs_' + c_hash + '.dat' 
    with ExtractedFeatures(X, feature_shp, batchsize, slm,
            feature_file_name) as features_fp:

        n_features = get_num_features(feature_shp, comparison)

        for split_id in range(num_splits):
            with PairFeatures(dataset, 'train_' + str(split_id), Xr,
                    n_features, features_fp, comparison,
                              train_pairs_filename) as train_Xy:
                with PairFeatures(dataset, 'test_' + str(split_id),
                        Xr, n_features, features_fp, comparison,
                                  test_pairs_filename) as test_Xy:
                    performances.append(train_classifier(config,
                                train_Xy, test_Xy, n_features))
    performance = np.array(performances).mean()
    result = dict(loss=performance, status='ok')
    
    if outfile is not None:
        outfh = open(outfile,'w')
        cPickle.dump(result, outfh)
        outfh.close()
    return result


def use_memmap(size):
    if size < 3e8:
        memmap = False
    else:
        memmap = True
    return memmap


class ExtractedFeatures(object):
    def __init__(self, X, feature_shp, batchsize, slm, filename):
        """
        X - 4-tensor of images
        feature_shp - 4-tensor of output feature shape (len matches X)
        batchsize - number of features to extract in parallel
        slm - feature-extraction module (with .process_batch() fn)
        filename - store features to memmap here

        returns - read-only memmap of features
        """

        size = 4 * np.prod(feature_shp)
        print('Total size: %i bytes (%.2f GB)' % (size, size / float(1e9)))
        memmap = use_memmap(size)   
        if memmap:
            print('Creating memmap %s for features of shape %s' % (
                                                  filename, str(feature_shp)))
            features_fp = np.memmap(filename,
                dtype='float32',
                mode='w+',
                shape=feature_shp)
        else:
            print('Using memory for features of shape %s' % str(feature_shp)) 
            features_fp = np.empty(feature_shp,dtype='float32')
    
        i = 0
        t0 = time.time()
        while True:
            if i + batchsize >= len(X):
                assert i < len(X)
                xi = np.asarray(X[-batchsize:])
                done = True
            else:
                xi = np.asarray(X[i:i+batchsize])
                done = False
            t1 = time.time()    
            feature_batch = slm.process_batch(xi)
            print('compute: ',time.time()-t1)
            t2 = time.time()
            delta = max(0,i + batchsize - len(X))
            features_fp[i:i+batchsize-delta] = feature_batch[delta:]
            print('write: ',time.time()-t2)
            if done:
                break

            i += batchsize
            if (i // batchsize) % 10 == 0:
                t_cur = time.time() - t0
                t_per_image = (time.time() - t0) / i
                t_tot = t_per_image * X.shape[0]
                print 'get_features_fp: %i / %i  mins: %.2f / %.2f ' % (
                        i , len(X),
                        t_cur / 60.0, t_tot / 60.0)
        # -- docs hers:
        #    http://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html
        #    say that deletion is the way to flush changes !?
        if memmap:
            del features_fp
            self.filename = filename
            self.features = np.memmap(filename,
                dtype='float32',
                mode='r',
                shape=feature_shp)
        else:
            self.filename = ''
            self.features = features_fp

    def __enter__(self):
        return self.features

    def __exit__(self, *args):
        if self.filename:
            os.remove(self.filename)


class PairFeatures(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def work(self, dataset, split, X, n_features,
             feature_fp, comparison, filename):
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
            feature_pairs_fp[ind] = compare(feature_fp[ai],
                                            feature_fp[bi],
                                            comparison)
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

class InvalidDescription(Exception):
    """Model description was invalid"""


def dict_add(a, b):
    rval = dict(a)
    rval.update(b)
    return rval

def get_into_shape(x):
    if hasattr(x,'__iter__'):
        x = np.array(x)
        assert x.ndim == 1
        x = x.reshape((1,len(x)))
    return x

def get_pythor_safe_description(description):
    description = copy.deepcopy(description)
    for layer_idx, layer_desc in enumerate(description):
        for (op_idx,(op_name, op_params)) in enumerate(layer_desc):
            if op_name.endswith('_h'):
                newname = op_name[:-2]
                layer_desc[op_idx] = (newname,op_params)
    return description

def get_comparison(config):
    comparison = config.get('comparison', 'concatenate')
    assert comparison in ['concatenate', 'add', 'sub', 'multiply', 'subsq']
    return comparison


def get_config_string(configs):
    return hashlib.sha1(repr(configs)).hexdigest()

def get_num_features(shp, comparison):
    """
    Given image features of size shp, how long many comparison features will
    there be?
    """
    if comparison == 'concatenate':
        return 2 * shp[1] * shp[2] * shp[3]
    elif comparison == 'add':
        return shp[1] * shp[2] * shp[3]
    elif comparison == 'sub':
        return shp[1] * shp[2] * shp[3]
    elif comparison == 'multiply':
        return shp[1] * shp[2] * shp[3]
    elif comparison == 'subsq':
        return shp[1] * shp[2] * shp[3]
    else:
        raise ValueError(comparison)


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


def compare(x, y, comparison):
    if comparison == 'concatenate':
        return np.concatenate([x.flatten(),y.flatten()])
    elif comparison == 'add':
        return x.flatten() + y.flatten()
    elif comparison == 'sub':
        return x.flatten() - y.flatten()
    elif comparison == 'multiply':
        return x.flatten() * y.flatten()
    elif comparison == 'subsq':
        return (x.flatten() - y.flatten()) ** 2
    else:
        raise ValueError(comparison)


def get_relevant_images(dataset, dtype='uint8'):
    # load & resize logic is LFW Aligned -specific
    assert 'Aligned' in str(dataset.__class__)

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

def flatten(x):
    return list(itertools.chain(*x))
    
    
#####heterogenous model stuff
def interpret_activ(filter, activ):
    n_filters = filter['initialize']['n_filters']
    generator = activ['generate'][0].split(':')
    vals = activ['generate'][1]
    
    if generator[0] == 'random':
        dist = generator[1]
        if dist == 'uniform':
            mean = vals['mean']
            delta = vals['delta']
            seed = vals['rseed']
            rng = np.random.RandomState(seed)
            low = mean-(delta/2)
            high = mean+(delta/2)
            size = (n_filters,)
            activ_vec = rng.uniform(low=low, high=high, size=size)
        elif dist == 'normal':
            mean = vals['mean']
            stdev = vals['stdev']
            seed = vals['rseed']
            rng = np.random.RandomState(seed)
            size = (n_filters,)
            activ_vec = rng.normal(loc=mean, scale=stdev, size=size)
        else:
            raise ValueError, 'distribution not recognized'
    elif generator[0] == 'fixedvalues':
        values = vals['values']
        num = n_filters / len(values)
        delta = n_filters - len(values)*num
        activ_vec = flatten([[v]*num for v in values] + [[values[-1]*delta]])
    else:
        raise ValueError, 'not recognized'
    
    return activ_vec
        
def interpret_model(desc):
    for layer in desc:
        for (opname,opparams) in layer:  
            if opname == 'fbcorr_h':
                kw = opparams['kwargs']
                if hasattr(kw.get('min_out'),'keys'):
                    kw['min_out'] = interpret_activ(opparams, kw['min_out'])
                if hasattr(kw.get('max_out'),'keys'):
                    kw['max_out'] = interpret_activ(opparams, kw['max_out'])                    
