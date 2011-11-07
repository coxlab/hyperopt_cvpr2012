import time

import numpy as np

import theano
import theano.tensor as tensor
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from pythor3.model.slm.plugins.passthrough.passthrough import (
        SequentialLayeredModelPassthrough,
        )
from pythor3.operation.lnorm_ import EPSILON as lnorm_EPSILON


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

        self._fn = theano.function([self.s_input], x,
                allow_input_downcast=True)

        if 0:
            theano.printing.debugprint(self._fn)

    def init_fbcorr(self, x, x_shp, n_filters,
            filter_shape,
            min_out=0,
            generate=None):
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
                border_mode='valid')
        x_shp = (x_shp[0], n_filters,
                x_shp[2] - filter_shape[0] + 1,
                x_shp[3] - filter_shape[1] + 1)
        return tensor.maximum(x, min_out), x_shp

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
            min_out=0,
            inker_shape=(3, 3),
            outker_shape=(3, 3),
            remove_mean=False,
            div_method='euclidean',
            threshold=0.,
            stretch=1.,
            mode='valid'):
        EPSILON = lnorm_EPSILON
        if outker_shape != inker_shape: raise NotImplementedError()
        if remove_mean: raise NotImplementedError()
        if div_method != 'euclidean': raise NotImplementedError()
        if mode != 'valid': raise NotImplementedError()
        if inker_shape != (3, 3): raise NotImplementedError()

        ssq, ssqshp = self.boxconv(x ** 2, x_shp, inker_shape, channels=True)
        arr_div = tensor.sqrt(ssq) + EPSILON
        if stretch != 1:
            xx = xx * stretch
            arr_div = arr_div * stretch
        denom = tensor.switch(arr_div < (threshold + EPSILON), 1.0, arr_div)

        xs = inker_shape[0] // 2
        ys = inker_shape[0] // 2
        arr_num = x[:, :, xs:-xs, ys:-ys]
        r = arr_num / denom
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


class LFWBandit(object):
    def __init__(self):
        pass

    @classmethod
    def evaluate(cls, config, ctrl):
        import skdata.lfw

        dataset = skdata.lfw.Funneled()

        X, y = dataset.img_classification_task()

        print X.shape

        batchsize = 16

        slm = TheanoSLM(
                in_shape=(batchsize, X.shape[3], X.shape[1], X.shape[2]),
                description=config['desc'])

        i = 0
        t0 = time.time()
        while True:
            xi = np.asarray(X[i:i+batchsize])
            yi = np.asarray(y[i:i+batchsize])
            if len(xi) == batchsize:
                slm.process_batch(xi.transpose(0, 3, 1, 2))
            else:
                break
            i += batchsize
            print 'i', i, xi.shape
            t_per_image = (time.time() - t0) / (i * batchsize)
            t_tot = t_per_image * X.shape[0]
            print 'feature_extraction_estimate', t_tot / 60.0, 'mins'
            assert i < X.shape[0]






