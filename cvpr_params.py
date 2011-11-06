

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
        kwargstr = ', '.join([str(x) + '=' + repr(v)] for k,v in self.kwargs) 
        
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
        

class gd(dict):
    def __repr__(self):
        return '{' + ', '.join(['"' + str(k) + '": ' + repr(v) for (k,v) in self.items()]) + '}'
        
class S(str):
    def __repr__(self):
        return '"' + self + '"' 


lnorm = {'inker_shape' : choice([3,5,7,9]),
         'outker_shape' : ref('this','inker_shape'),
         'remove_mean' : choice([0,1]),
         'stretch' : choice([.1,1,10]),
         'threshold' : choice([.1,1,10])
         }
          
lpool =  {'stride' : 2,
          'ker_shape' : choice([3,5,7,9]),
          'order' : choice([1,2,10])
         }


activ = {'min_out' : choice([null,0]),
         'max_out' : choice([1,null])
        }
        
           
filter1 = {'ker_shape' : choice([3,5,7,9]),
           'num_filters' : choice([16,32,64])}
            
filter2 = {'ker_shape' : choice([3,5,7,9]),
           'num_filters' : choice([16,32,64,128])}
                
filter3 = {'ker_shape' : choice([3,5,7,9]),
           'num_filters' : choice([16,32,64,128,256])}
            
layers = [{'lnorm' : lnorm},
          {'filter' : filter1,
           'activ' : activ,
           'lpool' : lpool,
           'lnorm' : lnorm},
          {'filter' : filter2,
           'activ' : activ,
           'lpool' : lpool,
           'lnorm' : lnorm},
          {'filter' : filter3,
           'activ' : activ,
           'lpool' : lpool,
           'lnorm' : lnorm}
         ]  

model = {'color_space' : "gray",
         'conv_mode': "valid",
         'preproc' : {'max_edge' : 150,
                      'lsum_ksize' : null,
                      'resize_method' : "bicubic",
                      'whiten' : choice([false,true])},
         'layers' : layers 

        }
          
          
    
### with activations -- replace activ in each case with:
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
       
layers_activ_hetero = [{'lnorm' : lnorm},
 					   {'filter' : filter1,
					    'activ' : choice([activ_uniform,
					                      activ_gaussian,
					                      activ_multiple_uniform]),
					    'lpool' : lpool,
					    'lnorm' : lnorm},
					   {'filter' : filter2,
					    'activ' :  choice([activ_uniform,
					                       activ_gaussian,
					                       activ_multiple_uniform]),
					    'lpool' : lpool,
					    'lnorm' : lnorm},
					   {'filter' : filter3,
					    'activ' :  choice([activ_uniform,
					                       activ_gaussian,
					                       activ_multiple_uniform]),
					    'lpool' : lpool,
					    'lnorm' : lnorm}
			 		  ] 
            

model_activ_hetero = {'color_space' : 'gray',
         'conv_mode': 'valid',
         'preproc' : {'max_edge' : 150,
                      'lsum_ksize' : null,
                      'resize_method' : 'bicubic',
                      'whiten' : choice([false,true])},
         'layers' : layers_activ_hetero
        }         