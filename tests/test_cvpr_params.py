import hyperopt.gdist as gd

import theano_slm
import cvpr_params


def test_cvprparams():
    template = gd.gDist( repr(cvpr_params.config).replace("'",'"'))
    config = template.sample(13)
    L = theano_slm.LFWBandit()
    L.evaluate(config,None)
    config = template.sample(10)
    L.evaluate(config,None)
    config = template.sample(2)
    L.evaluate(config,None)
