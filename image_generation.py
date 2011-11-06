import tempfile
import os
import itertools
import random
import urllib
import json

import numpy as np
import cairo

import renderer  # my 3d rendering stuff

from bson import SON

BASE_URL = 'http://50.19.109.25'
MODEL_URL = BASE_URL + ':9999/3dmodels?'
BG_URL =  BASE_URL + ':9999/backgrounds?'


class ImageConfigs(object):
    def __init__(self,config_gen_spec):
        self.configs = config_gen(self,config_gen_spec)
    
    def render_image(self,config,returnfh = False):
        return render_image(self,config,returnfh = returnfh)
         
def chain(iterables):
    for (ind,it) in enumerate(iterables):
        for element in it:
            yield ind,element

class config_gen(object):
    def __init__(self,IC,config):
        if not isinstance(config['images'],list):
            config['images'] = [config['images']]
        self.im_configs = config['images']
        param_list = []
        for I in config['images']:    
            if I['generator'] == 'renderman':
                newparams = reman_random_config_gen(IC,I)
            elif I['generator'] in ['lfw']:
                newparams = lfw_config_gen(IC,I)
            elif I['generator'] in ['caltech101', 'caltech256']:
                newparams = caltech_config_gen(IC,I)

            
            param_list.append(newparams)
            
        self.param_list = chain(param_list)

            
    def __iter__(self):
        return self
        
    def next(self):
        ind,x = self.param_list.next()
        x['image']['generator'] = self.im_configs[ind]['generator']
        return x
            
        
def lfw_config_gen(IC,config):
    pass
    

def caltech_config_gen(IC,config):
    pass
    

def renderman_random_config_gen(args):
    chooser = lambda v : (lambda : v[random.randint(0,len(v)-1)])    
    ranger = lambda v : (((chooser(np.arange(v['$gt'],v['$lt'],v['delta'])) if v.get('delta') else (lambda : (v['$lt'] - v['$gt']) * random.random() + v['$gt'])))  if isinstance(v,dict) else v) if v else None
    num = args['num_images']
    funcs = [(k,ranger(args.get(k))) for k in ['tx','ty','tz','rxy','rxz','ryz','sx','sy','sz','s','bg_phi','bg_psi']]

    if not 'model_ids' in args:
        models = json.loads(urllib.urlopen(MODEL_URL + 'action=distinct&field=id').read())
    else:
        models = args['model_ids']
    funcs1 = [('model_id',chooser(models))]
    if 'bg_ids' in args:
        bg_ids = args['bg_ids']
        funcs1.append(('bg_id',chooser(bg_ids)))
    elif 'bg_query' in args:
        bg_query = args['bg_query']
        bg_ids = json.loads(urllib.urlopen(BG_URL + 'query=' + json.dumps(bg_query) + '&distinct=path').read())
        funcs1.append(('bg_id',chooser(bg_ids)))
        
    if 'kenvs' in args:
        kenvs = args['kenvs']
        funcs1.append(('kenv',chooser(kenvs)))
    
    params = []
    for i in range(num):
        p = SON([])
        if args.get('use_canonical'):
            p['use_canonical'] = args['use_canonical']    
        for (k,f) in funcs + funcs1:
            if f:
                p[k] = f()
        if args.get('res'):
            p['res'] = args['res']

        params.append(SON([('image',p)]))
        
    return params



def render_image(IC,config,returnfh=False):
    generator = config['generator']
    if generator == 'renderman':
        return renderman_render(config,returnfh=returnfh)
    elif generator in ['lfw']:
        return render(IC,config)
    elif generator in ['caltech101', 'caltech256']:
        return caltech_render(IC,config)
    else:
        raise ValueError, 'image generator not recognized'


def get_canonical_view(m):
    v = json.loads(urllib.urlopen(MODEL_URL + 'query={"id":"' + m + '"}&fields=["canonical_view"]').read())[0]
    if v.get('canonical_view'):
        return v['canonical_view']
    
    
def renderman_render(config,returnfh = False):
    config = config.to_dict()
    
    params_list = [{}]
    param = params_list[0]
    if 'bg_id' in config:
        param['bg_id'] = config.pop('bg_id')
    if 'bg_phi' in config:
        param['bg_phi'] = config.pop('bg_phi')
    if 'bg_psi' in config:
        param['bg_phi'] = config.pop('bg_psi')
    if 'kenv' in config:
        param['kenv'] = config.pop('kenv')
    if 'res' in config:
        param['res_x'] = param['res_y'] = config['res']
    use_canonical = config.pop('use_canonical',False)
    if use_canonical:
        v = get_canonical_view(config['model_id'])
        if v:
            config['rotations'] = [{'rxy':v['rxy'],'rxz':v['rxz'],'ryz':v['ryz']},
                                   {'rxy':config.pop('rxy',0),'rxz':config.pop('rxz',0),'ryz':config.pop('ryz',0)}]
    param['model_params'] = [config]   

    orig_dir = os.getcwd()
    os.chdir(os.path.join(os.environ['HOME'] , 'render_wd'))
    tmp = tempfile.mkdtemp()
    renderer.render(tmp,params_list)
    imagefile = [os.path.join(tmp,x) for x in os.listdir(tmp) if x.endswith('.tif')][0]
    os.chdir(orig_dir)
     
    fh = open(imagefile)
    if returnfh:
        return fh
    else:
        return fh.read()
    

def lfw_render(config, returnfh = False):
    pass

def caltech_render(config, returnfh = False):
    pass

