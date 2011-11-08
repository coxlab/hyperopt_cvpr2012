import time
import os

import numpy as np

import theano
import theano.tensor as tensor
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from pythor3.model.slm.plugins.passthrough.passthrough import (
        SequentialLayeredModelPassthrough,
        )
from pythor3.operation import fbcorr_
from pythor3.operation import lnorm_


def dict_add(a, b):
    rval = dict(a)
    rval.update(b)
    return rval


class TheanoSLM(object):
    """
    SequentialLayeredModel clone implemented with Theano
    """

    def __init__(self, in_shape, description,
            dtype='float32', rng=888):

        if len(in_shape) == 2:
            self.in_shape = (1, 1,) +  in_shape
        elif len(in_shape) == 3:
            self.in_shape = (1,) + in_shape
        else:
            self.in_shape = in_shape
        assert len(self.in_shape) == 4
        print 'TheanoSLM.in_shape', self.in_shape

        # This guy is used to generate filterbanks
        self.SLMP = SequentialLayeredModelPassthrough(
                self.in_shape[2:],
                description,
                dtype=dtype)


        self.s_input = tensor.ftensor4('arr_in')
        self.rng = np.random.RandomState(rng)  # XXX check for rng being int

        x = self.s_input
        x_shp = self.in_shape
        for layer_idx, layer_desc in enumerate(description):
            for op_name, op_params in layer_desc:
                init_fn = getattr(self, 'init_' + op_name)
                x, x_shp = init_fn(x, x_shp,
                        **dict_add(
                            op_params.get('kwargs', {}),
                            op_params.get('initialize', {})))
                print 'added layer', op_name, 'shape', x_shp
        
        self.out_shape = x_shp

        self._fn = theano.function([self.s_input], x,
                allow_input_downcast=True)

        if 0:
            theano.printing.debugprint(self._fn)

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
        rval = tensor.reshape(
                conv.conv2d(x,
                    kerns,
                    image_shape=x_shp,
                    filter_shape=kerns.shape,
                    border_mode='valid'),
                rshp)
        return rval, rshp

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
            raise NotImplementedError('outker_shape != inker_shape')
        if stretch != 1:
            arr_num = arr_num * stretch
            arr_div = arr_div * stretch
        arr_div = tensor.switch(arr_div < (threshold + EPSILON), 1.0, arr_div)

        r = arr_num / arr_div
        r_shp = x_shp[0], x_shp[1], ssqshp[2], ssqshp[3]
        return r, r_shp

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

    def process_batch(self, arr_in):
        return self._fn(arr_in)

    def process(self, arr_in):
        if arr_in.ndim == 2:
            rval = self.process_batch(arr_in[None,None,:,:])[0]
            if rval.shape[0] > 1:
                # XXX: decide whether IO of self.fn is channel major or minor
                return rval.transpose(1, 2, 0)
            else:
                return rval[0]
        elif arr_in.ndim == 3:
            return self.process_batch(arr_in[None,:,:,:])[0]
        else:
            raise TypeError('rank error', arr_in)


import asgd
import numpy as np
import tempfile
import os.path as path
from early_stopping import fit_w_early_stopping, early_stopping

class LFWBandit(object):
    def __init__(self):
        pass

    @classmethod
    def evaluate(cls, config, ctrl):
        import skdata.lfw
        
        comparison = get_comparison(config)

        dataset = skdata.lfw.Funneled()

        X, y = dataset.img_classification_task()
        Xr, yr = dataset.raw_classification_task()
        Xr = np.array([os.path.split(xr)[-1] for xr in Xr])

        batchsize = 16

        slm = TheanoSLM(
                in_shape=(batchsize, X.shape[3], X.shape[1], X.shape[2]),
                description=config['desc'])

        outshape = slm.out_shape
    
        feature_shp = (X.shape[0],) + slm.out_shape[1:]
        features_fp = get_features_fp(X, feature_shp, batchsize, slm, 'features.dat')
        
        n_features = get_num_features(feature_shp, comparison)   
        print(n_features)

        clas = asgd.naive_asgd.NaiveBinaryASGD(n_features)
        
        num_splits = 1
        performances = []
        for split_id in range(num_splits):            
            A, B, ctrain = dataset.raw_verification_task_resplit(split='train_' + str(split_id))
            train_feature_pairs_fp = get_pair_fp(A, B, ctrain, Xr, 
                                                n_features, 'train_feature_pairs.dat', 
                                                features_fp, comparison, 'train_pairs.dat')
               
            A, B, ctest = dataset.raw_verification_task_resplit(split='test_' + str(split_id))
            test_feature_pairs_fp = get_pair_fp(A, B, ctest, Xr, 
                                                n_features, 'test_feature_pairs.dat', 
                                                features_fp, comparison, 'test_pairs.dat')      
                                        
        
            clas = fit_w_early_stopping(model=clas, es=early_stopping(warmup=20), 
                               train_X = train_feature_pairs_fp,
                               train_y = ctrain,
                               validation_X = test_feature_pairs_fp,
                               validation_y = ctest)
            
            prediction = clas.predict(test_feature_pairs_fp)
            
            performance = (prediction != ctest).astype(np.float).mean()
            performances.append(performances)

            test_feature_pairs_fp.close()
            os.remove(test_features_pairs_fp.filename)
            train_feature_pairs_fp.close()
            os.remove(train_features_pairs_fp.filename)
            
        performance = np.array(performances).mean()
        
        features_fp.close()
        os.remove(features_fp.filename)

        return dict(loss=performance, status='ok')


def get_features_fp(X, feature_shp, batchsize, slm, filename):
    #file = tempfile.NamedTemporaryFile(delete=False)
    print('TMP-->',filename)
    features_fp = np.memmap(filename,
                            dtype='float32',
                            mode='w+', 
                            shape=feature_shp)
                            
    i = 0
    t0 = time.time()                                
    while True:
        xi = np.asarray(X[i:i+batchsize])
        if len(xi) == batchsize:
            feature_batch = slm.process_batch(xi.transpose(0, 3, 1, 2))
            features_fp[i:i+batchsize] = feature_batch[:]                
        else:
            break
            
        i += batchsize
        print 'i', i, xi.shape
        t_per_image = (time.time() - t0) / (i * batchsize)
        t_tot = t_per_image * X.shape[0]
        print 'feature_extraction_estimate', t_tot / 60.0, 'mins'
        assert i < X.shape[0]
    
    del features_fp
    return np.memmap(filename,
                            dtype='float32',
                            mode='r', 
                            shape=feature_shp)    
            
def get_pair_fp(A, B, c, X, n_features, name, feature_fp, comparison, filename):
    Ar = np.array([os.path.split(ar)[-1] for ar in A])
    Br = np.array([os.path.split(br)[-1] for br in B])
    Aind = np.searchsorted(X, Ar)
    Bind = np.searchsorted(X, Br)        
    pair_shp = (len(c), n_features)        
    #file = tempfile.NamedTemporaryFile(delete=False)
    print('tmpfile',filename)
    feature_pairs_fp = np.memmap(filename,
                                dtype='float32',
                                mode='w+', 
                                shape=pair_shp)
    for (ind,(ai, bi)) in enumerate(zip(Aind,Bind)):
        feature_pairs_fp[ind] = compare(feature_fp[ai],
                                        feature_fp[bi],
                                        comparison)
        print(ind)
    del feature_pairs_fp
    return np.memmap(filename,
                                dtype='float32',
                                mode='r', 
                                shape=pair_shp)       


def get_comparison(config):
    comparison = config.get('comparison', 'concatenate')
    assert comparison in ['concatenate']
    return comparison


def get_num_features(x, comparison):
    if comparison == 'concatenate':
        return 2*x[1]*x[2]*x[3]


def compare(x, y, comparison):
    if comparison == 'concatenate':
        return np.concatenate([x.flatten(),y.flatten()])
