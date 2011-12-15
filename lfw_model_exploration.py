"""
# MUST be run with feature/separate_activation branch of thoreano
"""
import copy
import itertools

import numpy as np
import hyperopt.genson_bandits as gb
import pymongo as pm

import cvpr_params
from lfw import get_performance, DEFAULT_COMPARISONS

from cvpr_params import (null,
                         false,
                         true,
                         choice,
                         uniform,
                         gaussian,
                         lognormal,
                         qlognormal,
                         ref)


lnorm = {'kwargs':{'inker_shape' : choice([(3,3),(5,5),(7,7),(9,9)]),
         'outker_shape' : ref('this','inker_shape'),
         'remove_mean' : choice([0,1]),
         'stretch' : uniform(0,10),
         'threshold' : choice([null, uniform(0,10)])
         }}

lpool = {'kwargs': {'ker_shape' : choice([(3,3),(5,5),(7,7),(9,9)]),
          'order' : choice([1, 2, 10])
         }}

rescale = {'kwargs': {'stride' : 2}}

activ =  {'kwargs': {'min_out' : choice([null, 0]),
                     'max_out' : choice([1, null])}}

filter1 = dict(
         initialize=dict(
            filter_shape=choice([(3,3),(5,5),(7,7),(9,9)]),
            n_filters=choice([16, 32, 64]),
            generate=(
                'random:uniform',
                {'rseed': choice(range(5))})),
         kwargs={})

filter2 = copy.deepcopy(filter1)
filter2['initialize']['n_filters'] = choice([16, 32, 64, 128])
filter2['initialize']['generate'] = ('random:uniform', {'rseed': choice(range(5,10))})

filter3 = copy.deepcopy(filter1)
filter3['initialize']['n_filters'] = choice([16, 32, 64, 128, 256])
filter3['initialize']['generate'] = ('random:uniform', {'rseed': choice(range(10,15))})


###original model
original_params = {'desc': [[('lnorm', lnorm)],
            [('fbcorr', filter1),
             ('activ', activ),
             ('lpool', lpool),
             ('rescale', rescale),
             ('lnorm', lnorm)],
            [('fbcorr', filter2),
             ('activ', activ),
             ('lpool', lpool),
             ('rescale', rescale),
             ('lnorm', lnorm)],
            [('fbcorr', filter3),
             ('activ', activ),
             ('lpool', lpool),
             ('rescale', rescale),
             ('lnorm', lnorm)],
           ]}

class LFWBandit(gb.GensonBandit):
    def __init__(self):
        super(LFWBandit, self).__init__(source_string=cvpr_params.string(original_params))

    @classmethod
    def evaluate(cls, config, ctrl):
        result = get_performance(None, config)
        return result



######
order_choices = [[list(o[:ind]),list(o[ind:])] for o in list(itertools.permutations(['lpool','activ','lnorm'])) for ind in range(4)]
orders = choice(order_choices)
values = [{'lnorm':lnorm, 'lpool':lpool, 'activ':activ, 'filter':filter1},
          {'lnorm':lnorm, 'lpool':lpool, 'activ':activ, 'filter':filter2},
          {'lnorm':lnorm, 'lpool':lpool, 'activ':activ, 'filter':filter3}]
order_value_params = {'order':orders, 'values':values}

def get_model_config(config):
    before, after = config['order']
    values = config['values']
    layers = []
    config = {'desc':layers}
    for (layer_ind, vals) in enumerate(values):
        B = [(b, vals[b]) for b in before]
        A = [(a, vals[a]) for a in after]
        layer = B + [('fbcorr',vals['filter'])] + A + [('rescale',rescale)]
        layers.append(layer)
    return config

class LFWBanditModelExploration(gb.GensonBandit):
    def __init__(self):
        super(LFWBanditModelExploration, self).__init__(source_string=cvpr_params.string(order_value_params))

    @classmethod
    def evaluate(cls, config, ctrl, use_theano=True):
        config = get_model_config(config)
        result = get_performance(None, config)
        return result

def make_plot():
    conn = pm.Connection()
    db = conn['hyperopt']
    Jobs = db['jobs']
    exp_key0 = 'lfw_model_exploration.LFWBandit/hyperopt.theano_bandit_algos.TheanoRandom'
    H0 = 1-np.array([x['result']['loss'] for x in Jobs.find({'exp_key': exp_key0, 'state':2})])
    NF0 =  np.array([x['result']['data']['mult'][0]['n_features'] for x in Jobs.find({'exp_key': exp_key0, 'state':2})])
    
    exp_key = 'lfw_model_exploration.LFWBanditModelExploration/hyperopt.theano_bandit_algos.TheanoRandom'
    Qs = [{'exp_key':exp_key,'state':2,'spec.order':o} for o in order_choices]
    
    L = [1-np.array([x['result']['loss'] for x in Jobs.find(q,fields=['result.loss'])]) for q in Qs]
    NF = [np.array([x['result']['data']['mult'][0]['n_features'] for x in Jobs.find(q,fields=['result.data.mult'])]) for q in Qs]
        
    import matplotlib.pyplot as plt
    plt.figure()
    plt.boxplot([l-H0.mean() for l in L])
    plt.plot(range(1,len(L)+1),[np.mean(l) - H0.mean() for l in L],color='green')
    plt.scatter(range(1,len(L)+1),[np.mean(l) - H0.mean() for l in L])
    od = {'lpool': 'p', 'activ': 'a', 'lnorm': 'n'}
    order_labels = [','.join([od[b] for b in Before]) + '|' + ','.join([od[b] for b in After]) for (Before, After) in order_choices]
    plt.xticks(range(1,len(L)+1),order_labels, rotation=60)
    plt.title('Model form exploration')
    plt.ylabel('Performance relative to usual L3 on LFW')
    plt.xlabel('Architecture tag')
    plt.savefig('model_exploration_boxplots.png')
    

 
    return L, (H0, NF0)
