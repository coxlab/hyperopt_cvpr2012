import sys
import time
import os
import copy
import itertools
import tempfile
import os.path as path
import hashlib
import cPickle
import hashlib

import Image
import numpy as np

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
                        use_memmap)
from classifier import (train_classifier_normalize,
                        evaluate_classifier_normalize,
                        train_classifier,
                        evaluate_classifier)


DEFAULT_COMPARISONS = ['mult', 'sqrtabsdiff']

class LFWBandit(gb.GensonBandit):
    source_string = cvpr_params.string(cvpr_params.config)

    def __init__(self):
        super(LFWBandit, self).__init__(source_string=self.source_string)

    @classmethod
    def evaluate(cls, config, ctrl, use_theano=True, comparisons = DEFAULT_COMPARISONS):
        result = get_performance(None, config, use_theano=use_theano, comparisons=comparisons)
        return result


class LFWBanditSimpleArch(LFWBandit):
    source_string = cvpr_params.string(cvpr_params.simple_params)


class LFWBanditSimpleArch2(LFWBandit):
    source_string = cvpr_params.string(cvpr_params.simple_params2)


class LFWBanditUnidirectional(LFWBandit):
    source_string = cvpr_params.string(cvpr_params.uni_params)


class LFWBanditHetero(LFWBandit):
    source_string = cvpr_params.string(cvpr_params.config_h)


class LFWBanditHetero2(LFWBandit):
    source_string = cvpr_params.string(cvpr_params.config_h2)


class LFWBanditHetero3(LFWBandit):
    source_string = cvpr_params.string(cvpr_params.config_h3)


class LFWBanditHetero4(LFWBandit):
    source_string = cvpr_params.string(cvpr_params.config_h4)


class LFWBanditHetero5(LFWBandit):
    source_string = cvpr_params.string(cvpr_params.config_h5)


class LFWBanditHeteroTop5(LFWBandit):
    source_string = cvpr_params.string(cvpr_params.config_h_Top5).replace('u"','"').replace('None','null')

class LFWBanditHeteroTop5c(LFWBandit):
    source_string = cvpr_params.string(cvpr_params.config_h_Top5c).replace('u"','"').replace('None','null')

class LFWBanditHeteroTop(LFWBandit):
    source_string = cvpr_params.string(cvpr_params.config_h_top)

class LFWBanditHeteroPool(LFWBandit):
    source_string = cvpr_params.string(cvpr_params.config_h_pool)

class LFWBanditHeteroPool2(LFWBandit):
    source_string = cvpr_params.string(cvpr_params.config_h_pool2).replace('u"','"').replace('None','null')

class LFWBanditHeteroPool3(LFWBandit):
    source_string = cvpr_params.string(cvpr_params.config_h_pool3)

class LFWBanditHeteroPool4(LFWBandit):
    source_string = cvpr_params.string(cvpr_params.config_h_pool4)

class LFWBanditSGE(LFWBandit):
    @classmethod
    def evaluate(cls, config, ctrl, use_theano=True):
        outfile = os.path.join('/tmp',get_config_string(config))
        opstring = '-l qname=hyperopt.q -o /home/render/hyperopt_jobs -e /home/render/hyperopt_jobs'
        jobid = sge_utils.qsub(get_performance, (outfile, config, use_theano),
                     opstring=opstring)
        status = sge_utils.wait_and_get_statuses([jobid])
        return cPickle.loads(open(outfile).read())


def do_mod_details(params,mod_params):
    """dependency of do_mods
    """
    assert isinstance(params,dict) and isinstance(mod_params,dict)
    for k in mod_params:
        if k in params:
            if isinstance(params[k],dict):
                do_mod_details(params[k],mod_params[k])
            else:
                params[k] = mod_params[k]
        else:
            params[k] = mod_params[k]


def do_mods(spec, mods):
    """used to implement modifications from a "default base config" for the
    LFWBanditTopHetero bandit
    """
    for spec_l, mod_l in zip(spec, mods):
        print spec_l, mod_l
        for (ind,(opname, opparams)) in enumerate(mod_l):
            assert opname in [spec_l[ind][0], spec_l[ind][0] + '_h']
            do_mod_details(spec_l[ind][1], opparams)
            spec_l[ind][0] = opname


import pymongo
class LFWBanditTopHetero(LFWBandit):
    """bandit which looks up top bandits from a given run in the db and then runs
    heterogenous parameters around those top configs
    """
    source_string = cvpr_params.string(cvpr_params.config_mod)

    @classmethod
    def interpret_mod_config(cls, config):
        if not hasattr(cls,'conn'):
            cls.conn = pymongo.Connection()
            cls.db = cls.conn['hyperopt']
            cls.jobs = cls.db['jobs']
            cls.exp_key = 'lfw.LFWBandit/hyperopt.theano_bandit_algos.AdaptiveParzenGM'

        curs = cls.jobs.find({'exp_key':cls.exp_key, 'result.loss':{'$exists':True}}).sort('result.loss')

        top_val = config['top_model']
        desc = curs[top_val]['spec']['desc']

        do_mods(desc, config['mods'])

        return desc

    @classmethod
    def evaluate(cls, config, ctrl, use_theano=True):
        config = {'desc':cls.interpret_mod_config(config)}
        result = get_performance(None, config, use_theano=use_theano, comparisons=['mult'])
        return result


class LFWBanditEZSearch(gb.GensonBandit):
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

        config = {'desc' : layers, 'comparison' : comparison}
        source_string = repr(config).replace("'",'"')
        gb.GensonBandit.__init__(self, source_string=source_string)

    @classmethod
    def evaluate(cls, config, ctrl, use_theano=True):
        result = get_performance(None, config, use_theano=use_theano)
        return result


class LFWBanditEZSearch2(gb.GensonBandit):
    """
    This Bandit has the same evaluate function as LFWBandit,
    but the template is setup for more efficient search.

    Dan authored this class from LFWBanditEZSearch to add the possibility of
    random gabor initialization of first-layer filters
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
        activ =  {'min_out' : choice([null, 0]),
                  'max_out' : choice([1, null])}

        filter1 = dict(
                initialize=dict(
                    filter_shape=choice([(3,3),(5,5),(7,7),(9,9)]),
                    n_filters=qlognormal(np.log(32), 1, round=16),
                    generate=choice([('random:uniform',
                                     {'rseed': choice([11, 12, 13, 14, 15])}),
                                     ('random:gabor',
                                     {'min_wl': 2, 'max_wl': 20 ,
                                      'rseed': choice([11, 12, 13, 14, 15])})
                                     ])),
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

        config = {'desc' : layers, 'comparison' : comparison}
        source_string = repr(config).replace("'",'"')
        gb.GensonBandit.__init__(self, source_string=source_string)

    @classmethod
    def evaluate(cls, config, ctrl, use_theano=True):
        result = get_performance(None, config, use_theano=use_theano)
        return result




def test_splits():
    T = ['fold_' + str(i) for i in range(10)]
    splits = []
    for i in range(10):
        inds = range(10)
        inds.remove(i)
        v_ind = (i+1) % 10
        inds.remove(v_ind)
        test = T[i]
        validate = T[v_ind]
        train = [T[ind] for ind in inds]
        splits.append({'train': train,
                       'validate': validate,
                       'test': test})
    return splits


def get_test_performance(outfile, config, use_theano=True, flip_lr=False, comparisons=DEFAULT_COMPARISONS):
    """adapter to construct split notation for 10-fold split and call
    get_performance on it (e.g. this is like "test a config on View 2")
    """
    splits = test_splits()
    return get_performance(outfile, config, train_test_splits=splits,
                           use_theano=use_theano, flip_lr=flip_lr, tlimit=None,
                           comparisons=comparisons)


def get_performance(outfile, configs, train_test_splits=None, use_theano=True,
                    flip_lr=False, tlimit=35, comparisons=DEFAULT_COMPARISONS):
    """Given a config and list of splits, test config on those splits.

    Splits can either be "view1-like", e.g. train then test or "view2-like", e.g.
    train/validate and then test.  splits specified by a list of dictionaries
    with keys in ["train","validate","test"] and values are split names recognized
    by skdata.lfw.Aligned.raw_verification_task

    This function both extracts features AND runs SVM evaluation. See functions
    "get_features" and "train_feature"  below that split these two things up.
    At some point this function should be simplified by calls to those two.
    """
    import skdata.lfw
    c_hash = get_config_string(configs)
    if isinstance(configs, dict):
        configs = [configs]
    assert all([hasattr(comp_module,comparison) for comparison in comparisons])
    dataset = skdata.lfw.Aligned()
    if train_test_splits is None:
        train_test_splits = [{'train': 'DevTrain', 'test': 'DevTest'}]
    train_splits = [tts['train'] for tts in train_test_splits]
    validate_splits = [tts.get('validate',[]) for tts in train_test_splits]
    test_splits = [tts['test'] for tts in train_test_splits]
    all_splits = test_splits + validate_splits + train_splits
    X, y, Xr = get_relevant_images(dataset, splits = all_splits, dtype='float32')
    batchsize = 4
    performance_comp = {}
    feature_file_names = ['features_' + c_hash + '_' + str(i) +  '.dat' for i in range(len(configs))]
    train_pairs_filename = 'train_pairs_' + c_hash + '.dat'
    validate_pairs_filename = 'validate_pairs_' + c_hash + '.dat'
    test_pairs_filename = 'test_pairs_' + c_hash + '.dat'
    with TheanoExtractedFeatures(X, batchsize, configs, feature_file_names,
                                 use_theano=use_theano, tlimit=tlimit) as features_fps:

        feature_shps = [features_fp.shape for features_fp in features_fps]
        datas = {}
        for comparison in comparisons:
            print('Doing comparison %s' % comparison)
            perf = []
            datas[comparison] = []
            comparison_obj = getattr(comp_module,comparison)
            #how does tricks interact with n_features, if at all?
            n_features = sum([comparison_obj.get_num_features(f_shp) for f_shp in feature_shps])
            for tts in train_test_splits:
                print('Split', tts)
                if tts.get('validate') is not None:
                    train_split = tts['train']
                    validate_split = tts['validate']
                    test_split = tts['test']
                    with PairFeatures(dataset, train_split, Xr,
                            n_features, features_fps, comparison_obj,
                                      train_pairs_filename, flip_lr=flip_lr) as train_Xy:
                        with PairFeatures(dataset, validate_split,
                                Xr, n_features, features_fps, comparison_obj,
                                          validate_pairs_filename) as validate_Xy:
                            with PairFeatures(dataset, test_split,
                                Xr, n_features, features_fps, comparison_obj,
                                          test_pairs_filename) as test_Xy:
                                model, earlystopper, data, train_mean, train_std = \
                                                 train_classifier_normalize(train_Xy, validate_Xy)
                                print('earlystopper', earlystopper.best_y)
                                result = evaluate_classifier_normalize(model, test_Xy, train_mean, train_std)
                                perf.append(result['loss'])
                                print ('Split',tts, 'comparison', comparison, 'loss is', result['loss'])
                                n_test_examples = len(test_Xy[0])
                                result['split'] = tts
                                datas[comparison].append(result)

                else:
                    train_split = tts['train']
                    test_split = tts['test']
                    with PairFeatures(dataset, train_split, Xr,
                            n_features, features_fps, comparison_obj,
                                      train_pairs_filename, flip_lr=flip_lr) as train_Xy:
                        with PairFeatures(dataset, test_split,
                                Xr, n_features, features_fps, comparison_obj,
                                          test_pairs_filename) as test_Xy:
                            model, earlystopper, data, train_mean, train_std = \
                                                 train_classifier_normalize(train_Xy, test_Xy)
                            perf.append(data['loss'])
                            n_test_examples = len(test_Xy[0])
                            data['split'] = tts
                            datas[comparison].append(data)

            performance_comp[comparison] = float(np.array(perf).mean())

    performance = float(np.array(performance_comp.values()).min())
    result = dict(
            loss=performance,
            loss_variance=performance * (1 - performance) / n_test_examples,
            performances=performance_comp,
            data=datas,
            status='ok')

    if outfile is not None:
        outfh = open(outfile,'w')
        cPickle.dump(result, outfh)
        outfh.close()
    return result


def get_features(outfiles, configs, train_test_splits):
    """just extraction features.

    inputs are list of configs and a list of filenames
    to extract to.

    returns filehandles and names of all images extracted
    """
    arrays, labels, im_names = get_relevant_data(train_test_splits)

    batchsize = 4

    Ts = []
    for outfile, config in zip(outfiles,configs):
        T = TheanoExtractedFeatures(arrays, batchsize, [config], [outfile],
                                       tlimit=None, file_out = True)
        Ts.append(T)


    return Ts, im_names


def train_features(infiles, inshapes, im_names, train_test_splits,
                   flip_lr=False,
                   comparisons=DEFAULT_COMPARISONS,
                   n_jobs=False,
                   outfile=None,
                   trace_normalize=True):

    """just train.  From features in "infiles" input.  Also needs list of all
       image names, inshapes, and train_test_splits.   This is annoying and
       should perhaps be removed.
       Parallelizes on splits using joblib  (specify number of jobs via n_jobs)
    """
    assert all([hasattr(comp_module,comparison) for comparison in comparisons])

    datas = {}
    from joblib import Parallel, delayed
    g = (delayed(train_features_single)(infiles, inshapes, im_names, tts, comp, flip_lr=flip_lr, trace_normalize=trace_normalize) for comp in comparisons for tts in train_test_splits)
    R = Parallel(n_jobs=n_jobs,verbose=1)(g)

    ind = 0
    performance_comp = {}
    for comparison in comparisons:
        perf = []
        datas[comparison] = []
        for tts in train_test_splits:
            result, n_test_examples = R[ind]
            result['split'] = tts
            perf.append(result['loss'])
            datas[comparison].append(result)
            ind += 1
        performance_comp[comparison] = float(np.array(perf).mean())

    performance = float(np.array(performance_comp.values()).min())

    Result = dict(
            loss=performance,
            loss_variance=performance * (1 - performance) / n_test_examples,
            performances=performance_comp,
            data=datas,
            status='ok')

    if outfile is not None:
        outfh = open(outfile,'w')
        cPickle.dump(Result, outfh)
        outfh.close()
    return Result


def train_features_single(infiles, inshapes, im_names, tts, comparison, flip_lr=False, trace_normalize=True):

    import skdata.lfw
    dataset = skdata.lfw.Aligned()

    arrays = get_arrays(infiles, inshapes)
    comparison_obj = getattr(comp_module,comparison)
    n_features = sum([comparison_obj.get_num_features(f_shp) for f_shp in inshapes])

    print('Split', tts)
    if tts.get('validate') is not None:
        train_split = tts['train']
        validate_split = tts['validate']
        test_split = tts['test']
        with PairFeatures(dataset, train_split, im_names,
                n_features, arrays, comparison_obj,
                          None, flip_lr=flip_lr) as train_Xy:
            with PairFeatures(dataset, validate_split,
                    im_names, n_features, arrays, comparison_obj,
                              None) as validate_Xy:
                with PairFeatures(dataset, test_split,
                    im_names, n_features, arrays, comparison_obj,
                              None) as test_Xy:
                    train_Xy, validate_Xy, test_Xy, m, s, m1 = normalize((train_Xy,
                                                            validate_Xy,
                                                            test_Xy),
                                                            obs_norm=obs_norm)
                    model, earlystopper, data = train_classifier(train_Xy, validate_Xy)
                    print('earlystopper', earlystopper.best_y)
                    result = evaluate_classifier(model, test_Xy)
                    print ('Split',tts, 'comparison', comparison, 'loss is', result['loss'])
                    n_test_examples = len(test_Xy[0])
    else:
        train_split = tts['train']
        test_split = tts['test']
        with PairFeatures(dataset, train_split, im_names,
                n_features, arrays, comparison_obj,
                          None, flip_lr=flip_lr) as train_Xy:
            with PairFeatures(dataset, test_split,
                    im_names, n_features, arrays, comparison_obj,
                              None) as test_Xy:
                train_Xy, test_Xy, m, s, m1 = normalize((train_Xy, test_Xy), trace_normalize=trace_normalize)
                model, earlystopper, result = train_classifier(train_Xy, test_Xy)
                n_test_examples = len(test_Xy[0])

    return result, n_test_examples


def normalize(feats_Xy, trace_normalize=True):
    """Performs normalizations before training on a list of feature array/label
    pairs. first feature array in list is taken by default to be training set
    and norms are computed relative to that one.
    """
    feats, labels = zip(*feats_Xy)
    train_f = feats[0]
    m = train_f.mean(axis=0)
    s = np.maximum(train_f.std(axis=0), 1e-8)
    feats = [(f - m) / s for f in feats]
    train_f = feats[0]
    m1 = np.maximum(np.sqrt((train_f**2).sum(axis=1)).mean(), 1e-8)
    if trace_normalize:
        feats = [f / m1 for f in feats]
    feats_Xy = tuple(zip(feats,labels))
    return feats_Xy + (m, s, m1)



######utils

class PairFeatures(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def work(self, dset, split, X, n_features,
             features_fps, comparison_obj, filename, flip_lr=False):
        if isinstance(split, str):
            split = [split]
        A = []
        B = []
        labels = []
        for s in split:
            if s.startswith('re'):
                s = s[2:]
                A0, B0, labels0 = dset.raw_verification_task_resplit(split=s)
            else:
                A0, B0, labels0 = dset.raw_verification_task(split=s)
            A.extend(A0)
            B.extend(B0)
            labels.extend(labels0)
        Ar = np.array([os.path.split(ar)[-1] for ar in A])
        Br = np.array([os.path.split(br)[-1] for br in B])
        labels = np.array(labels)
        Aind = np.searchsorted(X, Ar)
        Bind = np.searchsorted(X, Br)
        assert len(Aind) == len(Bind)
        pair_shp = (len(labels), n_features)


        if flip_lr:
            pair_shp = (4 * pair_shp[0], pair_shp[1])

        size = 4 * np.prod(pair_shp)
        print('Total size: %i bytes (%.2f GB)' % (size, size / float(1e9)))
        memmap = filename is not None and use_memmap(size)
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
        feature_labels = []

        for (ind,(ai, bi)) in enumerate(zip(Aind, Bind)):
            # -- this flattens 3D features to 1D features
            if flip_lr:
                feature_pairs_fp[4 * ind + 0] = np.concatenate(
                        [comparison_obj(
                            fp[ai, :, :, :],
                            fp[bi, :, :, :])
                            for fp in features_fps])
                feature_pairs_fp[4 * ind + 1] = np.concatenate(
                        [comparison_obj(
                            fp[ai, :, ::-1, :],
                            fp[bi, :, :, :])
                            for fp in features_fps])
                feature_pairs_fp[4 * ind + 2] = np.concatenate(
                        [comparison_obj(
                            fp[ai, :, :, :],
                            fp[bi, :, ::-1, :])
                            for fp in features_fps])
                feature_pairs_fp[4 * ind + 3] = np.concatenate(
                        [comparison_obj(
                            fp[ai, :, ::-1, :],
                            fp[bi, :, ::-1, :])
                            for fp in features_fps])

                feature_labels.extend([labels[ind]] * 4)
            else:
                feats = [comparison_obj(fp[ai],fp[bi])
                        for fp in features_fps]
                feature_pairs_fp[ind] = np.concatenate(feats)
                feature_labels.append(labels[ind])
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

        self.labels = np.array(feature_labels)


    def __enter__(self):
        self.work(*self.args, **self.kwargs)
        return (self.features, self.labels)

    def __exit__(self, *args):
        if self.filename:
            os.remove(self.filename)



def get_config_string(configs):
    return hashlib.sha1(repr(configs)).hexdigest()


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


def unroll(X):
    Y = []
    for x in X:
        if isinstance(x,str):
            Y.append(x)
        else:
            Y.extend(x)
    return np.unique(Y)


def get_arrays(filenames, inshapes):
    return [np.memmap(filename,
                    dtype='float32',
                    mode='r',
                    shape=inshape) for filename,inshape in zip(filenames, inshapes)]


def get_relevant_data(train_test_splits):
    import skdata.lfw

    dataset = skdata.lfw.Aligned()

    train_splits = [tts['train'] for tts in train_test_splits]
    validate_splits = [tts.get('validate',[]) for tts in train_test_splits]
    test_splits = [tts['test'] for tts in train_test_splits]

    all_splits = test_splits + validate_splits + train_splits

    return get_relevant_images(dataset, splits = all_splits, dtype='float32')


def get_relevant_images(dataset, splits=None, dtype='uint8'):
    # load & resize logic is LFW Aligned -specific
    assert 'Aligned' in str(dataset.__class__)


    Xr, yr = dataset.raw_classification_task()
    Xr = np.array(Xr)

    if splits is not None:
        splits = unroll(splits)

    if splits is not None:
        all_images = []
        for s in splits:
            if s.startswith('re'):
                A, B, c = dataset.raw_verification_task_resplit(split=s[2:])
            else:
                A, B, c = dataset.raw_verification_task(split=s)
            all_images.extend([A,B])
        all_images = np.unique(np.concatenate(all_images))

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



def get_config_string(configs):
    return hashlib.sha1(repr(configs)).hexdigest()


def random_id():
    return hashlib.sha1(str(np.random.randint(10,size=(32,)))).hexdigest()
