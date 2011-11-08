import copy

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
           
filter1 = {'initialize': {'filter_shape' : choice([(3,3),(5,5),(7,7),(9,9)]),
					   'n_filters' : choice([16,32,64]),
					   'generate': ('random:uniform',{'rseed':42})},
		   'kwargs': activ}
           
filter2 = {'initialize': {'filter_shape' : choice([(3,3),(5,5),(7,7),(9,9)]),
					   'n_filters' : choice([16,32,64,128]),
					   'generate': ('random:uniform',{'rseed':42})},
		   'kwargs': activ}
					      
filter3 = {'initialize': {'filter_shape' : choice([(3,3),(5,5),(7,7),(9,9)]),
					   'n_filters' : choice([16,32,64,128,256]),
					   'generate': ('random:uniform',{'rseed':42})},
		   'kwargs': activ}					      
            
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


config = {'desc' : layers}

          
          
    
### with activations -- replace activ in each case with:
###TODO:  get this in the right format for pythor
activ_uniform = {'min_out' : choice([null, {'dist' : 'uniform',
                                            'mean' : uniform(-.2,.2),
                                            'delta' : uniform(0, .2)}]),
                 'max_out' : choice([null, {'dist' : 'uniform',
                                            'mean' : uniform(1-.2,1+.2),
                                            'delta' : uniform(0, .2)}])                                          
                }
              
activ_gaussian = {'min_out' : choice([null, {'dist' : 'gaussian',
                                                      'mean' : uniform(-.2,.2),
                                                      'stdev' : uniform(0, .2)}]),
                  'max_out' : choice([null, {'dist' : 'gaussian',
                                                      'mean' : uniform(1-.2,1+.2),
                                                      'stdev' : uniform(0, .2)}])                                          
                 }
                 
       
                 
def activ_multiple_gen(dist,minargs,minkwargs,maxargs,maxkwargs,num):
    activ = {}
    activ['min_out'] = choice([null] + [{'values': [dist(*minargs,**minkwargs) for ind in range(k)]} for k in range(num)])
    activ['max_out'] = choice([null] + [{'values': [dist(*maxargs,**maxkwargs) for ind in range(k)]} for k in range(num)]) 
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
           

        