import copy

def string(s):
    return repr(s).replace("'",'"')

##Plain vanilla params  -- from FG11 paper

class Null(object):
    def __repr__(self):
        return 'null'
null = Null()

class FALSE(object):
    def __repr__(self):
        return 'false'
false = FALSE()

class TRUE(object):
    def __repr__(self):
        return 'true'
true = TRUE()


def repr(x):
    if isinstance(x,str):
        return '"' + str(x) + '"'
    else:
        return x.__repr__()

class gObj(object):
    def __init__(self,*args,**kwargs):
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        argstr = ', '.join([repr(x) for x in self.args])
        kwargstr = ', '.join([str(k) + '=' + repr(v) for k,v in self.kwargs.items()])

        astr = argstr + (', ' + kwargstr if kwargstr else '')
        return self.name + '(' + astr + ')'


class choice(gObj):
    name = 'choice'

class uniform(gObj):
    name = 'uniform'

class gaussian(gObj):
    name = 'gaussian'

class lognormal(gObj):
    name = 'lognormal'

class qlognormal(gObj):
    name = 'qlognormal'

class ref(object):
    def __init__(self,*p):
        self.path = p

    def __repr__(self):
        return '.'.join(self.path)


lnorm = {'kwargs':{'inker_shape' : choice([(3,3),(5,5),(7,7),(9,9)]),
         'outker_shape' : ref('this','inker_shape'),
         'remove_mean' : choice([0,1]),
         'stretch' : choice([.1,1,10]),
         'threshold' : choice([.1,1,10])
         }}

lpool = {'kwargs': {'stride' : 2,
          'ker_shape' : choice([(3,3),(5,5),(7,7),(9,9)]),
          'order' : choice([1,2,10])
         }}


activ =  {'min_out' : choice([null,0]),
          'max_out' : choice([1,null])}

filter1 = dict(
        initialize=dict(
            filter_shape=choice([(3,3),(5,5),(7,7),(9,9)]),
            n_filters=choice([16,32,64]),
            generate=(
                'random:uniform',
                {'rseed': choice([11, 12, 13, 14, 15])})),
         kwargs=activ)

filter2 = dict(
        initialize=dict(
            filter_shape=choice([(3, 3), (5, 5), (7, 7), (9, 9)]),
            n_filters=choice([16, 32, 64, 128]),
            generate=(
                'random:uniform',
                {'rseed': choice([21, 22, 23, 24, 25])})),
         kwargs=activ)

filter3 = dict(
        initialize=dict(
            filter_shape=choice([(3, 3), (5, 5), (7, 7), (9, 9)]),
            n_filters=choice([16, 32, 64, 128, 256]),
            generate=(
                'random:uniform',
                {'rseed': choice([31, 32, 33, 34, 35])})),
         kwargs=activ)

layers = [[('lnorm', lnorm)],
          [('fbcorr', filter1),
           ('lpool', lpool),
           ('lnorm', lnorm)],
          [('fbcorr', filter2),
           ('lpool' , lpool),
           ('lnorm' , lnorm)],
          [('fbcorr', filter3),
           ('lpool', lpool),
           ('lnorm', lnorm)]
         ]


comparison = ['mult', 'absdiff', 'sqrtabsdiff', 'sqdiff']

config = {'desc' : layers, 'comparison' : comparison}


activ_uniform = {'min_out' : choice([null, {'generate' : ('random:uniform',
                                                          {'rseed':42,
                                                           'mean':uniform(-.2,.2),
                                                           'delta':uniform(0,.2)}),

                                            }]),
                 'max_out' : choice([null, {'generate' : ('random:uniform',
                                                          {'rseed':42,
                                                           'mean':uniform(.8,1.2),
                                                           'delta':uniform(0,.2)}),

                                            }]),
                }

activ_gaussian = {'min_out' : choice([null, {'generate' : ('random:normal',
                                                          {'rseed':42,
                                                           'mean':uniform(-.2,.2),
                                                           'stdev':uniform(0,.2)}),

                                            }]),
                 'max_out' : choice([null, {'generate' : ('random:normal',
                                                          {'rseed':42,
                                                           'mean':uniform(.8,1.2),
                                                           'stdev':uniform(0,.2)}),

                                            }]),
                }



def activ_multiple_gen(dist,minargs,minkwargs,maxargs,maxkwargs,num):
    activ = {}
    activ['min_out'] = choice([null] + [{'generate': ('fixedvalues',{'values':[dist(*minargs,**minkwargs) for ind in range(k)]})} for k in range(1,num+1)])
    activ['max_out'] = choice([null] + [{'generate': ('fixedvalues',{'values':[dist(*maxargs,**maxkwargs) for ind in range(k)]})} for k in range(1,num+1)])
    return activ

activ_multiple_uniform = activ_multiple_gen(uniform,(-.2,.2),{},(1-.2,1+.2),{},5)

filter1_h = copy.deepcopy(filter1)
filter1_h['kwargs'] = choice([activ_uniform,
                            activ_gaussian,
                            activ_multiple_uniform])

filter2_h = copy.deepcopy(filter2)
filter2_h['kwargs'] = choice([activ_uniform,
                            activ_gaussian,
                            activ_multiple_uniform])

filter3_h = copy.deepcopy(filter3)
filter3_h['kwargs'] = choice([activ_uniform,
                            activ_gaussian,
                            activ_multiple_uniform])

layers_h = [[('lnorm', lnorm)],
            [('fbcorr_h', filter1_h),
             ('lpool', lpool),
             ('lnorm', lnorm)],
            [('fbcorr_h', filter2_h),
             ('lpool' , lpool),
             ('lnorm' , lnorm)],
            [('fbcorr_h', filter3_h),
             ('lpool', lpool),
             ('lnorm', lnorm)]
           ]


config_h = {'desc' : layers_h, 'comparison' : comparison}


activ_uniform2 = {'min_out' : choice([null, {'generate' : ('random:uniform',
                                                           {'rseed':42,
                                                           'mean':uniform(-.5,.1),
                                                           'delta':uniform(0,.2)}),

                                            }]),
                 'max_out' : choice([null, 1]),
                }

filter1_h2 = copy.deepcopy(filter1)
filter1_h2['kwargs'] = activ_uniform2

filter2_h2 = copy.deepcopy(filter2)
filter2_h2['kwargs'] = activ_uniform2

filter3_h2 = copy.deepcopy(filter3)
filter3_h2['kwargs'] = activ_uniform2


layers_h2 = [[('lnorm', lnorm)],
            [('fbcorr_h', filter1_h2),
             ('lpool', lpool),
             ('lnorm', lnorm)],
            [('fbcorr_h', filter2_h2),
             ('lpool' , lpool),
             ('lnorm' , lnorm)],
            [('fbcorr_h', filter3_h2),
             ('lpool', lpool),
             ('lnorm', lnorm)]
           ]


config_h2 = {'desc' : layers_h2, 'comparison' : comparison}


activ_uniform3 = {'min_out' : choice([null, {'generate' : ('random:uniform',
                                                           {'rseed':42,
                                                           'mean':0,
                                                           'delta':uniform(0,.3)}),

                                            }]),
                 'max_out' : choice([null, 1]),
                }

filter1_h3 = copy.deepcopy(filter1)
filter1_h3['kwargs'] = activ_uniform3

filter2_h3 = copy.deepcopy(filter2)
filter2_h3['kwargs'] = activ_uniform3

filter3_h3 = copy.deepcopy(filter3)
filter3_h3['kwargs'] = activ_uniform3


layers_h3 = [[('lnorm', lnorm)],
            [('fbcorr_h', filter1_h3),
             ('lpool', lpool),
             ('lnorm', lnorm)],
            [('fbcorr_h', filter2_h3),
             ('lpool' , lpool),
             ('lnorm' , lnorm)],
            [('fbcorr_h', filter3_h3),
             ('lpool', lpool),
             ('lnorm', lnorm)]
           ]

config_h3 = {'desc' : layers_h3, 'comparison' : comparison}

activ_uniform4 = {'min_out' : choice([null, {'generate' : ('random:uniform',
                                                           {'rseed':42,
                                                           'mean':-0.2,
                                                           'delta':uniform(0,.3)}),

                                            }]),
                 'max_out' : choice([null, 1]),
                }

filter1_h4 = copy.deepcopy(filter1)
filter1_h4['kwargs'] = activ_uniform4

filter2_h4 = copy.deepcopy(filter2)
filter2_h4['kwargs'] = activ_uniform4

filter3_h4 = copy.deepcopy(filter3)
filter3_h4['kwargs'] = activ_uniform4


layers_h4 = [[('lnorm', lnorm)],
            [('fbcorr_h', filter1_h4),
             ('lpool', lpool),
             ('lnorm', lnorm)],
            [('fbcorr_h', filter2_h4),
             ('lpool' , lpool),
             ('lnorm' , lnorm)],
            [('fbcorr_h', filter3_h4),
             ('lpool', lpool),
             ('lnorm', lnorm)]
           ]


config_h4 = {'desc' : layers_h4, 'comparison' : comparison}

activ_uniform5 = {'min_out' : choice([null, uniform(-.4,.2)]),
                 'max_out' : choice([null, 1]),
                }

filter1_h5 = copy.deepcopy(filter1)
filter1_h5['kwargs'] = activ_uniform5

filter2_h5 = copy.deepcopy(filter2)
filter2_h5['kwargs'] = activ_uniform5

filter3_h5 = copy.deepcopy(filter3)
filter3_h5['kwargs'] = activ_uniform5


layers_h5 = [[('lnorm', lnorm)],
            [('fbcorr_h', filter1_h5),
             ('lpool', lpool),
             ('lnorm', lnorm)],
            [('fbcorr_h', filter2_h5),
             ('lpool' , lpool),
             ('lnorm' , lnorm)],
            [('fbcorr_h', filter3_h5),
             ('lpool', lpool),
             ('lnorm', lnorm)]
           ]


config_h5 = {'desc' : layers_h5, 'comparison' : comparison}


activ_uniform_top = ref('root','activ')

filter1_h_top = copy.deepcopy(filter1)
filter1_h_top['kwargs'] = activ_uniform_top

filter2_h_top = copy.deepcopy(filter2)
filter2_h_top['kwargs'] = activ_uniform_top

filter3_h_top = copy.deepcopy(filter3)
filter3_h_top['kwargs'] = activ_uniform_top


layers_h_top = [[('lnorm', lnorm)],
            [('fbcorr_h', filter1_h_top),
             ('lpool', lpool),
             ('lnorm', lnorm)],
            [('fbcorr_h', filter2_h_top),
             ('lpool' , lpool),
             ('lnorm' , lnorm)],
            [('fbcorr_h', filter3_h_top),
             ('lpool', lpool),
             ('lnorm', lnorm)]
           ]

config_h_top = {'desc' : layers_h_top,'comparison' : comparison, 'activ': activ_uniform3}

import cPickle

from Top5 import Top5

Top5a = copy.deepcopy(Top5)
for t in Top5a:
    t[1][0][1]['kwargs']['min_out'] = {'generate' : ('random:uniform',
                                                           {'rseed':42,
                                                           'mean':0,
                                                           'delta':uniform(0,.3)})}
    t[1][0][0] = 'fbcorr_h'
config_h_Top5 = {'desc': Top5a[0], 'comparison' : comparison}


lpool_h = {'kwargs': {'stride' : 2,
          'ker_shape' : choice([(3,3),(5,5),(7,7),(9,9)]),
          'order' : {'generate':('fixedvalues',{'values':[1,2,10]})}
         }}

layers_h_pool = [[('lnorm', lnorm)],
            [('fbcorr', filter1),
             ('lpool_h', lpool_h),
             ('lnorm', lnorm)],
            [('fbcorr', filter2),
             ('lpool' , lpool),
             ('lnorm' , lnorm)],
            [('fbcorr', filter3),
             ('lpool', lpool),
             ('lnorm', lnorm)]
           ]

config_h_pool = {'desc': layers_h_pool, 'comparison' : comparison}

lpool_h2 =  {'generate':('fixedvalues',{'values':choice([[1],[2],[4],[10],[1,2],[1,10],[1,2,10],[1,2,4,10],[2,10]])})}

lpool_h3 = copy.deepcopy(lpool_h)
lpool_h3['order'] = lpool_h2

Top5b = copy.deepcopy(Top5)
for t in Top5b:
    t[1][1][1]['kwargs']['order'] = lpool_h2
    t[1][1][0] = 'lpool_h'
config_h_pool2 = {'desc': choice(Top5b), 'comparison' : comparison}


layers_h_pool3 = [[('lnorm', lnorm)],
            [('fbcorr', filter1),
             ('lpool_h', lpool_h3),
             ('lnorm', lnorm)],
            [('fbcorr', filter2),
             ('lpool' , lpool),
             ('lnorm' , lnorm)],
            [('fbcorr', filter3),
             ('lpool', lpool),
             ('lnorm', lnorm)]
           ]

config_h_pool3 = {'desc': layers_h_pool3, 'comparison' : comparison}

lpool_h4o =  {'generate':('fixedvalues',{'values':choice([[uniform(1,5)],[uniform(1,5),
                                                                          uniform(1,5)],
                                                                         [uniform(1,5),uniform(1,5),uniform(1,5)]])})}

lpool_h4 = copy.deepcopy(lpool_h)
lpool_h4['order'] = lpool_h4o

layers_h_pool4 = [[('lnorm', lnorm)],
            [('fbcorr', filter1),
             ('lpool_h', lpool_h4),
             ('lnorm', lnorm)],
            [('fbcorr', filter2),
             ('lpool' , lpool),
             ('lnorm' , lnorm)],
            [('fbcorr', filter3),
             ('lpool', lpool),
             ('lnorm', lnorm)]
           ]

config_h_pool4 = {'desc': layers_h_pool4, 'comparison' : comparison}


Top5c = copy.deepcopy(Top5)
for t in Top5c:
    t[1][0][1]['kwargs']['min_out'] = {'generate' : ('random:uniform',
                                                           {'rseed':42,
                                                           'mean':0,
                                                           'delta':uniform(0,.3)})}
    t[1][0][0] = 'fbcorr_h'
    t[2][0][1]['kwargs']['min_out'] = {'generate' : ('random:uniform',
                                                           {'rseed':42,
                                                           'mean':0,
                                                           'delta':uniform(0,.3)})}
    t[2][0][0] = 'fbcorr_h'
    t[3][0][1]['kwargs']['min_out'] = {'generate' : ('random:uniform',
                                                           {'rseed':42,
                                                           'mean':0,
                                                           'delta':uniform(0,.3)})}
    t[3][0][0] = 'fbcorr_h'
config_h_Top5c = {'desc': choice(Top5c), 'comparison' : comparison}



lpool_mod = {'kwargs': {'order' : {'generate':('fixedvalues',{'values':choice([[uniform(1,10)],
                                                                               [uniform(1,10),
                                                                                uniform(1,10)],
                                                                               [uniform(1,10),
                                                                                uniform(1,10),
                                                                                uniform(1,10)]
                                                                               ])
                                                             })}
                        }
            }

fbcorr_mod = {'kwargs': {'min_out' : {'generate' : ('random:uniform',
                                                           {'rseed':42,
                                                           'mean':uniform(-.5,.1),
                                                           'delta':uniform(0,.2)})
                                     }
                        }
              }


lnorm_mod = {'kwargs':{'stretch' : lognormal(0, 1),
                       'threshold' : lognormal(0, 1)
                      }
            }

config_mod = {'top_model': choice(range(10)),
     'mods' : [[('lnorm',lnorm_mod)],
               [('fbcorr_h', fbcorr_mod),
                ('lpool_h', lpool_mod),
                ('lnorm',lnorm_mod)],
               [('fbcorr_h', fbcorr_mod),
                ('lpool_h', lpool_mod),
                ('lnorm',lnorm_mod)],
               [('fbcorr_h', fbcorr_mod),
                ('lpool_h', lpool_mod),
                ('lnorm',lnorm_mod)]
               ]

    }


##################simple architecture

rescale = {'kwargs': {'stride' : 2}}

filter1_simple = dict(
        initialize=dict(
            filter_shape=choice([(3,3),(5,5),(7,7),(9,9)]),
            n_filters=choice([16,32,64]),
            generate1=(
                'random:uniform',
                {'rseed': choice([11, 12, 13, 14, 15])}),
            generate2=(
                'random:uniform',
                {'rseed': choice([0, 1, 2, 3, 4])}),
            exp1=choice([1, 2, 4, 10]),
            exp2=choice([1, 2, 4, 10]),
                ),
         kwargs=activ)

filter2_simple = copy.deepcopy(filter1_simple)
filter2_simple['initialize']['n_filters'] = choice([16,32,64,128])

filter3_simple = copy.deepcopy(filter1_simple)
filter3_simple['initialize']['n_filters'] = choice([16,32,64,128,256])

simple_layers = [[('lnorm', lnorm)],
          [('fbcorr2', filter1_simple),
           ('rescale', rescale)],
          [('fbcorr2', filter2_simple),
           ('rescale', rescale)],
          [('fbcorr2', filter3_simple),
           ('rescale', rescale)],
         ]

simple_params = {'desc' : simple_layers}

lpool1 = {'kwargs': {'stride' : 2,
          'ker_shape' : (1,1),
          'order' : 1
         }}

simple_layers2 = [[('lnorm', lnorm)],
          [('fbcorr', filter1),
           ('lpool', lpool1)],
          [('fbcorr', filter2),
           ('lpool' , lpool1)],
          [('fbcorr', filter3),
           ('lpool', lpool1)]
         ]

simple_params2 = {'desc' : simple_layers2}

lpool_simple_2 = {'kwargs': {'stride' : 2,
          'ker_shape' : (3,3),
          'order' : 2
         }}

simple_layers3 = [[('lnorm', lnorm)],
          [('fbcorr', filter1),
           ('lpool', lpool_simple_2)],
          [('fbcorr', filter2),
           ('lpool' , lpool_simple_2)],
          [('fbcorr', filter3),
           ('lpool', lpool_simple_2)]
         ]
         
simple_params3 = {'desc' : simple_layers3}

simple_params3_reorder = {'desc':
          [[('lnorm', lnorm)],
          [('lpool', lpool_simple_2),
           ('fbcorr', filter1)],
          [('lpool' , lpool_simple_2),
           ('fbcorr', filter2)],
          [('lpool', lpool_simple_2),
           ('fbcorr', filter3)]
          ]}
         

simple_params4 = {'desc':
          [[('lnorm', lnorm)],
          [('fbcorr', filter1),
           ('lpool', lpool)],
          [('fbcorr', filter2),
           ('lpool' , lpool)],
          [('fbcorr', filter3),
           ('lpool', lpool)]
         ]}
         
         
lpool_s1 = {'kwargs': {'stride' : 1,
          'ker_shape' : choice([(3,3),(5,5),(7,7),(9,9)]),
          'order' : choice([1,2,10])
         }}

lpool_s2 = {'kwargs': {'stride' : 2,
          'ker_shape' : (1,1),
          'order' : 1
         }}
         
params_reorder = {'desc':
          [[('lnorm', lnorm),
           ('lpool', lpool_s1),
           ('fbcorr', filter1),
           ('lpool', lpool_s2),
           ],
          [('lnorm', lnorm),
           ('lpool', lpool_s1),
           ('fbcorr', filter2),
           ('lpool', lpool_s2),
           ],
          [('lnorm', lnorm),
           ('lpool', lpool_s1),
           ('fbcorr', filter3),
           ('lpool', lpool_s2),
           ]] 
         }

params_reorder2 = {'desc':
          [[('lpool', lpool_s1),
           ('lnorm', lnorm),
           ('fbcorr', filter1),
           ('lpool', lpool_s2),
           ],
          [('lpool', lpool_s1),
           ('lnorm', lnorm),
           ('fbcorr', filter2),
           ('lpool', lpool_s2),
           ],
          [('lpool', lpool_s1),
           ('lnorm', lnorm),
           ('fbcorr', filter3),
           ('lpool', lpool_s2),
           ]] 
         }

filter1_uni = copy.deepcopy(filter1)
filter1_uni['initialize']['generate'] = ('unidirectional',{'normalize':false})

filter2_uni = copy.deepcopy(filter2)
filter2_uni['initialize']['generate'] = ('unidirectional',{'normalize':false})

filter3_uni = copy.deepcopy(filter3)
filter3_uni['initialize']['generate'] = ('unidirectional',{'normalize':false})

layers_uni = [[('lnorm', lnorm)],
          [('fbcorr', filter1_uni),
           ('lpool', lpool),
           ('lnorm', lnorm)],
          [('fbcorr', filter2_uni),
           ('lpool' , lpool),
           ('lnorm' , lnorm)],
          [('fbcorr', filter3_uni),
           ('lpool', lpool),
           ('lnorm', lnorm)]
         ]
uni_params = {'desc' : layers_uni}

#######

filter1_gabor = copy.deepcopy(filter1)
filter1_gabor['initialize']['generate'] = ('random:gabor',
                                          {'min_wl': 2, 'max_wl': 20 ,
                                           'rseed': choice([11, 12, 13, 14, 15])})

layers_gabor = [[('lnorm', lnorm)],
          [('fbcorr', filter1_gabor),
           ('lpool', lpool),
           ('lnorm', lnorm)],
          [('fbcorr', filter2),
           ('lpool' , lpool),
           ('lnorm' , lnorm)],
          [('fbcorr', filter3),
           ('lpool', lpool),
           ('lnorm', lnorm)]
         ]
gabor_params = {'desc' : layers_gabor}
