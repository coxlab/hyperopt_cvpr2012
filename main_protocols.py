import sys
import cPickle
import hashlib
import os

import pymongo as pm
import gridfs
from bson import SON

from starflow.protocols import protocolize, actualize
from starflow.utils import activate

import image_generation as rendering
import model_generation

from dbutils import get_config_string, get_filename, createCertificateDict

DB_NAME = 'cvpr_2012'


#################IMAGE_PREPARATION#############
#################IMAGE_PREPARATION#############
#################IMAGE_PREPARATION#############
#################IMAGE_PREPARATION#############
#################IMAGE_PREPARATION#############

def image_protocol_hash(config_path):
    config = get_config(config_path)
    image_hash = get_config_string(config['images'])
    return image_hash

def image_protocol(config_path,write = False,parallel=False,reads=None):
    config = get_config(config_path) 
    image_hash = image_protocol_hash(config_path)
    image_certificate = '../.image_certificates/' + image_hash
    if reads is None:
        reads = ()
        
    if  not parallel:
        D = [('generate_images',generate_images,(image_certificate,image_hash,config,reads))]
    else:
        D = [('generate_images',generate_images_parallel,(image_certificate,image_hash,config))]
    
    if write:
        actualize(D)
    return D,image_hash

@activate(lambda x : x[3], lambda x : x[0])    
def generate_images(outfile,im_hash,config_gen,reads):

    conn = pm.Connection(document_class = SON)
    db = conn[DB_NAME]
    im_coll = db['images.files']
    im_fs = gridfs.GridFS(db,'images')
    
    remove_existing(im_coll,im_fs,im_hash)
    
    IC = rendering.ImageConfigs(config_gen)
        
    for (i,x) in enumerate(IC.configs):
        if (i/100)*100 == i:
            print(i,x)       
        if x['image']['generator'] != 'dataset_api':
            image_string = IC.render_image(x['image']) 
        else:
            image_string = open(x['image'].pop('img_fullpath')).read()
        y = SON([('config',x)])
        filename = get_filename(x)
        y['filename'] = filename
        y['__hash__'] = im_hash
        im_fs.put(image_string,**y)
                
        
    createCertificateDict(outfile,{'image_hash':im_hash,'args':config_gen})

def generate_and_insert_single_image(x,im_hash):
    
    conn = pm.Connection(document_class = SON)
    db = conn[DB_NAME]
    im_coll = db['images.files']
    im_fs = gridfs.GridFS(db,'images')
    
    image_string = rendering.render_image(None,x['image']) 
    y = SON([('config',x)])
    filename = get_filename(x)
    y['filename'] = filename
    y['__hash__'] = im_hash
    im_fs.put(image_string,**y)


@activate(lambda x : (), lambda x : x[0])    
def generate_images_parallel(outfile,im_hash,config_gen):

    conn = pm.Connection(document_class = SON)
    db = conn[DB_NAME]
    im_coll = db['images.files']
    im_fs = gridfs.GridFS(db,'images')
    
    remove_existing(im_coll,im_fs,im_hash)
    
    IC = rendering.ImageConfigs(config_gen)
       
    jobids = []
    for (i,x) in enumerate(IC.configs):
        jobid = qsub(generate_and_insert_single_image,(x,im_hash),opstring='-pe orte 2 -l qname=rendering.q -o /home/render/image_jobs -e /home/render/image_jobs')  
        jobids.append(jobid)
        
    createCertificateDict(outfile,{'image_hash':im_hash,'args':config_gen})

    return {'child_jobs':jobids}
    


#################MODEL_GENERATION#############
#################MODEL_GENERATION#############
#################MODEL_GENERATION#############
#################MODEL_GENERATION#############
#################MODEL_GENERATION#############
    
def model_protocol_hash(config_path):
    config = get_config(config_path)
    model_hash = get_config_string(config['models'])
    return model_hash
 
 
def model_protocol(config_path,write = False,parallel=False):

    config = get_config(config_path)
    model_hash = model_protocol_hash(config_path)
    model_certificate = '../.model_certificates/' + model_hash

    D = [('generate_models',generate_models,(model_certificate,model_hash,config))]
    
    if write:
        actualize(D)
    return D,model_hash


@activate(lambda x : (), lambda x : x[0])    
def generate_models(outfile,m_hash,config_gen):

    conn = pm.Connection(document_class = SON)
    db = conn[DB_NAME]
    m_coll = db['models.files']
    m_fs = gridfs.GridFS(db,'models')
    
    remove_existing(m_coll,m_fs,m_hash)
    
    M = model_generation.model_config_generator(config_gen)       
    
    for (i,m) in enumerate(M):
        if isinstance(m['model'],list):
            filterbanks = [model_generation.get_model(model) for model in m['model']]
        else:
            filterbanks = model_generation.get_model(m['model'])
            
        filterbank_string = cPickle.dumps(filterbanks)
        if (i/5)*5 == i:
            print(i,m) 
        
        y = SON([('config',m)])
        filename = get_filename(m)
        y['filename'] = filename
        y['__hash__'] = m_hash
        m_fs.put(filterbank_string,**y)
        
    createCertificateDict(outfile,{'model_hash':m_hash,'args':config_gen})
    
    


#################OPTIMIZATION###########
#################OPTIMIZATION###########
#################OPTIMIZATION###########
#################OPTIMIZATION###########
#################OPTIMIZATION###########


def optimization_protocol(opt_config_path,model_config_path,image_config_path,
                      convolve_func_name='numpy', write=False,parallel=False):
    model_config_gen = get_config(model_config_path)
    model_hash = get_config_string(model_config_gen['models'])
    model_certificate = '../.model_certificates/' + model_hash
    
    image_config_gen = get_config(image_config_path)
    image_hash =  get_config_string(image_config_gen['images'])
    image_certificate = '../.image_certificates/' + image_hash

    opt_config = get_config(opt_config_path)
    opt_config = opt_config.pop('optimization')

    D = []
    DH = {}
    for opt in opt_config:
        overall_config_gen = SON([('models',model_config_gen['models']),('images',image_config_gen['images']),('optimization',task)])
        opt_hash = get_config_string(overall_config_gen)    
        
        optimization_certificate = '../.optimization_certificates/' + opt_hash
        func = optimize
                                                
        op = ('optimization_' + opt_hash,func, (optimization_certificate,
                                              image_certificate,
                                              model_certificate,
                                              opt_config_path,
                                              convolve_func_name,
                                              opt,
                                              opt_hash))                                                
        D.append(op)
        DH[opt_hash] = [op]
             
    if write:
        actualize(D)
    return DH

@activate(lambda x : x[0], lambda x : (x[1],x[2],x[3]))
def optimize(outfile,
			  image_certificate,
			  model_certificate,
			  opt_config_path,
			  convolve_func_name,
			  opt,
			  opt_hash):
	
    image_config_gen, model_hash, image_hash = prepare_hyperopt(opt_hash,image_certificate_file, model_certificate_file,opt)
                                                                                                  
    #put args for bandit in file
    
    (tmpfile,tmpfilename) = tempfile.mkstemp()
    source_string = opt['source_string']
    bandit = opt['bandit']
    bandit_algo = opt['bandit_algo']
    steps = opt['steps']
    bandit_args = opt.get('bandit_args',{'args':(),'kwargs':{}})
    bandit_algo_args = opt.get('bandit_algo_args',{'args':(),'kwargs':{}})
    
    args = (source_string,
                 image_hash,
                 model_hash,
                 image_config_gen,
                 opt_hash,
                 convolve_func_name) + bandit_args['args']
    kwargs = bandit_args['kwargs']
    argd = {'args':args,'kwargs':kwargs}
    cPickle.dump(argd,tmpfile)
    tmpfile.close()
  
    (tmpfile,tmpfilename2) = tempfile.mkstemp()
    cPickle.dump(bandit_algo_args,tmpfile)
    tmpfile.close()
    
    command = 'hyperopt-mongo-search --block --steps ' + str(steps) + ' --bandit-argfile ' + tmpfilename + ' --bandit-algo-argfile ' + tmpfilename2 + ' ' + bandit + ' ' + bandit_algo
    status = os.system(command)

    if not status == 0:
        raise ValueError, 'Command %s threw an exception, see %s for log.' % (command, tmpfilename)
    else:
        os.remove(tmpfilename)
    
    createCertificateDict(outfile,{'image_file':image_certificate_file,'models_file':model_certificate_file})
                                                                                

def prepare_hyperopt():
    pass
    
         
#####utils


def remove_existing(coll,fs, hash):
    existing = coll.find({'__hash__':hash})
    for e in existing:
        fs.delete(e['_id'])


def get_config(config_fname):
    config_path = os.path.abspath(config_fname)
    print("Config file:", config_path)
    config = {}
    execfile(config_path, {},config)
    
    return config['config']