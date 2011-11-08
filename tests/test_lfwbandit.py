import hyperopt.gdist

from theano_slm import TheanoSLM, LFWBandit
import cvpr_params

def run_lfw(seed):
    template = hyperopt.gdist.gDist(
            repr(cvpr_params.config).replace("'",'"'))
    config = template.sample(seed)
    bandit = LFWBandit()
    config['workdir'] = '/mnt/bottom/tmp'
    config['cvpr_params_seed'] = seed
    bandit.evaluate(config, None)

def test_seed_A():
    run_lfw(31)

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
