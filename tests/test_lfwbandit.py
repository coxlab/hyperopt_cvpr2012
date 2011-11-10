import sys
import hyperopt.gdist

from hyperopt_cvpr2012.theano_slm import TheanoSLM, LFWBandit, InvalidDescription
from hyperopt_cvpr2012.theano_slm import LFWBanditEZSearch
from hyperopt_cvpr2012 import cvpr_params

def run_lfw(seed, use_theano=True, compute_features=True):
    template = hyperopt.gdist.gDist(
            repr(cvpr_params.config).replace("'",'"'))
    config = template.sample(seed)
    bandit = LFWBandit()
    config['cvpr_params_seed'] = seed
    config['compute_features'] = compute_features
    result = bandit.evaluate(config, None, use_theano=use_theano)
    print ('FINAL result for seed=%i' % seed), result

def test_many_seeds(start=0, stop=100):
    for seed in range(start, stop):
        try:
            result = run_lfw(seed)
        except InvalidDescription:
            print 'INVALID SEED', seed
        sys.stdout.flush()
        sys.stderr.flush()
    assert 0  # make sure nosetests prints everything out

def test_many_more_seeds():
    return test_many_seeds(1000, 1100)

def test_seed_A():
    run_lfw(31)

def test_seed_10():
    run_lfw(10)

def test_seed_A_pythor():
    run_lfw(31, use_theano=False)

def test_genson_sampling_not_random():
    template = hyperopt.gdist.gDist(
            repr(cvpr_params.config).replace("'",'"'))
    # -- test that template.sample(seed) is a deterministic function of seed
    for seed in range(20):
        a = template.sample(seed)
        # print a
        for i in range(10):
            b = template.sample(seed)
            assert a == b


def test_genson_sampling_not_random_more():
    template = hyperopt.gdist.gDist(
            repr(cvpr_params.config_h).replace("'",'"'))
    # -- test that template.sample(seed) is a deterministic function of seed
    for seed in range(20):
        a = template.sample(seed)
        # print a
        for i in range(10):
            b = template.sample(seed)
            assert a == b


def test_ezsearch():
    bandit = LFWBanditEZSearch()
    config = bandit.template.sample(58)
    result = bandit.evaluate(config, None, use_theano=True)
    print ('FINAL result for seed=%i' % seed), result

