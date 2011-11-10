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


activ_uniform_2 = {'min_out' : choice([null, {'generate' : ('random:uniform',
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

filter2_h2 = copy.deepcopy(filter2)
filter2_h2['kwargs'] = activ_uniform2
					    				        
       
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
