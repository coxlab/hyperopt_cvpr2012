from classifier import train_classifier, split_center_normalize
from theano_slm import FeatureExtractor, slm_from_config, InvalidDescription
from utils import TooLongException, RGBImgLoaderResizer

def raw_folds(dataset, K, n_per_fold, seed):
    """
    Returns list of class-balanced index lists
    """
    meta = dataset.meta
    search_order = np.random.RandomState(seed).permutation(len(meta))
    examples = {}
    next_pos = {}
    for name in dataset.names:
        examples[name] = [ii for ii in search_order if meta[ii]['name'] == name]
        next_pos[name] = 0

    jj = 0
    folds = []
    for k in xrange(K):
        folds.append([])
        for ii in xrange(n_per_fold):
            name_ii = dataset.names[jj % len(dataset.names)]
            folds[k].append(examples[name_ii][next_pos[name_ii]])
            next_pos[name_ii] += 1
            jj += 1
    return folds

def classif_from_inds(dataset, inds):
    image_paths = [dataset.meta[ind]['filename'] for ind in inds]
    names = [dataset.meta[ind]['name'] for ind in inds]
    labels = np.searchsorted(dataset.names, names)
    return image_paths, np.asarray(labels)


# XXX: ERROR: this returns balanced folds such that the entire set of folds
# contains n_train + n_test from each class.
# XXX XXX
# THIS IS NOT A STANDARD TRAINING PROTOCOL FOR CALTECH 256
def raw_train_test(dataset, split, n_splits,
        seed=123,
        n_train_per_class=15,
        n_test_per_class=15):

    n_required = (n_train_per_class + n_test_per_class) * len(dataset.names)

    folds = raw_folds(dataset, n_splits,
            n_per_fold=int(np.ceil(float(n_required) / n_splits)),
            seed=seed)

    test_inds = folds[split]
    train_inds = []
    for ii, fold in enumerate(folds):
        if ii != split:
            train_inds.extend(fold)

    train_names, train_y = classif_from_inds(dataset, train_inds)
    test_names, test_y = classif_from_inds(dataset, test_inds)

    return train_names, train_y, test_names, test_y

def raw_train_test_valid(dataset, split, n_splits,
        seed=123,
        n_train_per_class=15,
        n_test_per_class=15):

    n_required = (n_train_per_class + n_test_per_class) * len(dataset.names)

    folds = raw_folds(dataset, n_splits,
            n_per_fold=int(np.ceil(float(n_required) / n_splits)),
            seed=seed)

    vsplit = (split + 1) % n_splits

    test_inds = folds[split]
    valid_inds = folds[vsplit]
    train_inds = []
    for ii, fold in enumerate(folds):
        if ii != split and ii != vsplit:
            train_inds.extend(fold)

    train_names, train_y = classif_from_inds(dataset, train_inds)
    test_names, test_y = classif_from_inds(dataset, test_inds)
    valid_names, valid_y = classif_from_inds(dataset, valid_inds)

    return train_names, train_y, test_names, test_y, valid_names, valid_y

class CaltechBandit(gb.GensonBandit):
    source_string = cvpr_params.string(cvpr_params.config)

    def __init__(self):
        super(CaltechBandit, self).__init__(source_string=self.source_string)

    @classmethod
    def evaluate(cls, config, ctrl, use_theano=True, TEST=False):
        dataset = getattr(skdata.caltech, cls.dset_name)()
        c_hash = get_config_string(config)

        num_splits = 10  # XXX: this has to match dataset right?
        batchsize = 4
        classif_batchsize=10

        arrays, paths = get_relevant_images(dataset, cls.img_shape, 'float32')

        n_classes = len(dataset.names)
        feature_file_name = 'features_' + c_hash + '.dat'
        try:
            slm = slm_from_config(config, arrays.shape, batchsize=4)
        except InvalidDescription:
            return dict(status='ok',
                    loss=1.0,
                    msg='invalid document')

        fext = FeatureExtractor(arrays, slm,
                filename=feature_file_name,   # filename means to use memmap
                #tlimit=60,                    # minutes
                TEST=TEST)

        with fext as features:
            features_shp = features.shape
            perfs = []
            splits_to_try = [0]
            for split_id in splits_to_try:
                train_names, train_y, test_names, test_y = raw_train_test(
                        dataset,
                        split=split_id,
                        n_splits=10)

                train_inds = np.searchsorted(paths, train_names)
                test_inds =  np.searchsorted(paths, test_names)

                # copy train data into memory
                train_X = features[train_inds]
                train_X.shape = (len(train_X), np.prod(features.shape[1:]))

                # shuffle training data so all classes are approximately same
                # in test and valid
                # TODO: ensure exact class balance
                np.random.RandomState(123).shuffle(train_X)
                np.random.RandomState(123).shuffle(train_y)

                # compute mean, std, and normalize features all in-place
                fit_Xy, valid_Xy, train_mean, train_std = split_center_normalize(
                        train_X, train_y,
                        inplace=True,
                        batchsize=classif_batchsize)


                # TODO: stepsize should be sampled? (in config)

                model, earlystopper, data = train_classifier(fit_Xy, valid_Xy,
                        verbose=True,
                        step_sizes=[1e-4],
                        use_theano=True,
                        batchsize=classif_batchsize)
                print 'best y', earlystopper.best_y
                print 'best time', earlystopper.best_time

                # free training data from memory
                del train_X

                # load test data into memory
                test_X = features[test_inds]
                test_X.shape = (len(test_X), np.prod(features.shape[1:]))

                # normalize test_X in-place
                test_X -= train_mean
                test_X /= train_std

                pred_y = model.predict(test_X)
                test_err = (pred_y != test_y).astype('float').mean()
                print 'test error', test_err

                perfs.append(dict(
                    valid_err=float(earlystopper.best_y),
                    test_err=float(test_err)))

        performance = float(np.mean([p['valid_err'] for p in perfs]))
        result = dict(
            loss=performance,
            perfs=perfs,
            status='ok')

        return result


class Caltech101Bandit(CaltechBandit):
    dset_name = 'Caltech101'
    img_shape = (200, 200, 3)


class Caltech256Bandit(CaltechBandit):
    dset_name = 'Caltech256'
    img_shape = (200, 200, 3)


class Caltech256Bandit_100x100(Caltech256Bandit):
    img_shape = (128, 128, 3)


def get_config_string(configs):
    return hashlib.sha1(repr(configs)).hexdigest()


def get_relevant_images(dataset, shape, dtype):
    # Often not all images are used by the splits, so that error scores are
    # calculated based on balanced classes

    X, yr = dataset.raw_classification_task()
    Xr = np.array(X)

    dsets = []
    for ind in range(dataset.num_splits):
        Xtr, _ytr, Xte, _yte = raw_train_test(dataset, split=ind,
                n_splits=10)
        dsets.extend([Xtr, Xte])
    all_images = np.unique(np.concatenate(dsets))

    inds = np.searchsorted(Xr, all_images)
    Xr = Xr[inds]
    yr = yr[inds]

    rows, cols, channels = shape
    assert channels == 3

    arrays = skdata.larray.lmap(
                RGBImgLoaderResizer(
                    shape=(rows, cols),
                    dtype=dtype),
                Xr)

    return arrays, Xr


def main_run_debug():
    _, _cmd, seed, TEST = sys.argv
    bandit = Caltech256Bandit()
    config = bandit.template.sample(int(seed))
    result = bandit.evaluate(config, ctrl=None, TEST=bool(int(TEST)))
    print '"""""'
    print "RESULT"
    print result


def main_run_debug_100():
    _, _cmd, seed, TEST = sys.argv
    bandit = Caltech256Bandit_100x100()
    config = bandit.template.sample(int(seed))
    result = bandit.evaluate(config, ctrl=None)
    print '"""""'
    print "RESULT"
    print result


def main_features_from_seed():
    _, cmd, seed, filename = sys.argv

    dataset = skdata.caltech.Caltech256()
    bandit = Caltech256Bandit()
    arrays, paths = get_relevant_images(dataset, bandit.img_shape, 'float32')
    config = bandit.template.sample(int(seed))
    fext = FeatureExtractor(arrays,
            slm_from_config(config, arrays.shape, batchsize=4),
            filename=filename,
            TEST=False)
    fext.extract_to_memmap()


def main_classify_features():
    _, cmd, in_filename, use_theano, stepsize = sys.argv

    in_dtype, in_shape = cPickle.load(open(in_filename + '.info'))
    features = np.memmap(in_filename, dtype=in_dtype, mode='r', shape=in_shape)
    print('loaded features of shape %s' % str(features.shape))

    dataset = skdata.caltech.Caltech256()
    _arrays, paths = get_relevant_images(dataset,
            shape=(200, 200), # doesn't matter because we don't use _arrays
            dtype='float32')

    split_id = 0
    batchsize = 10
    train_names, train_y, test_names, test_y = raw_train_test(dataset,
            split=split_id, n_splits=10)

    train_inds = np.searchsorted(paths, train_names)
    test_inds =  np.searchsorted(paths, test_names)

    train_X = features[train_inds]   # -- copy into memory
    train_X.shape = (len(train_X), np.prod(features.shape[1:]))

    np.random.RandomState(123).shuffle(train_X)
    np.random.RandomState(123).shuffle(train_y)

    test_X = features[test_inds]     # -- copy into memory
    test_X.shape = (len(test_X), np.prod(features.shape[1:]))

    # compute mean, std, and normalize features all in-place
    fit_Xy, valid_Xy, train_mean, train_std = split_center_normalize(
            train_X, train_y, inplace=True, batchsize=batchsize)
    test_X -= train_mean
    test_X /= train_std

    model, earlystopper, data = train_classifier(fit_Xy, valid_Xy,
            verbose=True,
            step_sizes=[float(stepsize)],
            use_theano=(use_theano == 'use_theano'),
            batchsize=batchsize)
    print 'best y', earlystopper.best_y
    print 'best time', earlystopper.best_time
    pred_y = model.predict(test_X)
    print 'test error', (pred_y != test_y).astype('float').mean()


if __name__ == '__main__':
    cmd = sys.argv[1]
    main = globals()['main_' + cmd]
    sys.exit(main())

