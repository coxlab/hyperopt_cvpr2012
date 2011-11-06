import random
import multiprocessing
import functools 
import copy
import math
import sys
import cPickle
import hashlib
import os
import scipy as sp
import numpy as np

from dbutils import get_config_string, get_filename, reach_in, createCertificateDict, son_escape, get_most_recent_files
from sge_utils import qsub

import pymongo as pm
import gridfs
import bson
from bson import SON

from starflow.sge_utils import wait_and_get_statuses

from starflow.utils import ListUnion, uniqify
from processing import image2array, preprocess, postprocess
import traintest
import svm
import v1like_funcs as v1f

import pythor3.operation.lnorm_.plugins.cthor
import pythor3.operation.lnorm_.plugins.numpy_naive
import pythor3.operation.lpool_.plugins.cthor
import pythor3.operation.lpool_.plugins.numpy_naive
import pythor3.operation.fbcorr_.plugins.cthor
import pythor3.operation.fbcorr_.plugins.numpyFFT

from pythor3.operation.fbcorr_ import (
    DEFAULT_STRIDE,
    DEFAULT_MIN_OUT,
    DEFAULT_MAX_OUT,
    DEFAULT_MODE)

try:
    import pycuda.driver as cuda
except:
    GPU_SUPPORT = False
else:
    import pythor3.operation.fbcorr_.plugins.cuFFT as cuFFT
    GPU_SUPPORT = True


FEATURE_CACHE = {}

DB_NAME = 'cvpr_2012'

def get_performance(model,task,image_hash,model_hash,image_config_gen,opt_hash):

    connection = pm.Connection(document_class = SON)
    db = connection['thor']
    perf_col = db['performance']
    split_fs = gridfs.GridFS(db, 'splits')
    splitperf_fs = gridfs.GridFS(db, 'split_performance')
    model_hash = 'optimization'
    
    splits = generate_splits(task,image_hash,'images') 
    for (ind,split) in enumerate(splits):
        put_in_split(split,image_config_gen,model,task,
                     opt_hash,ind,split_fs)
        print('evaluating split %d' % ind)
        res = extract_and_evaluate_core(split,model,convolve_func_name,
                                        task,use_db=use_db)    
        put_in_split_result(res,image_config_gen,model,task,
                            opt_hash,ind,splitperf_fs)
        split_results.append(res)
    put_in_performance(split_results,image_config_gen,model,
                                  model_hash,image_hash,
                                  perf_col,task,opt_hash)

def extract_and_evaluate_core(split,m,convolve_func_name,task,cache_port,use_db = False):
    classifier_kwargs = task.get('classifier_kwargs',{})  
    train_data = split['train_data']
    test_data = split['test_data']
    train_labels = split['train_labels']
    test_labels = split['test_labels']                
    train_filenames = [t['filename'] for t in train_data]
    test_filenames = [t['filename'] for t in test_data]
    assert set(train_filenames).intersection(test_filenames) == set([])

    existing_train_features = [get_from_cache((tf,m,task.get('transform_average')),FEATURE_CACHE) for tf in train_filenames]
    existing_train_labels = [train_labels[i] for (i,x) in enumerate(existing_train_features) if x is not None]
    existing_train_filenames = [train_filenames[i] for (i,x) in enumerate(existing_train_features) if x is not None]
    new_train_filenames = [train_filenames[i] for (i,x) in enumerate(existing_train_features) if x is None]
    reordered_train_filenames = existing_train_filenames + new_train_filenames
    new_train_labels = [train_labels[i] for (i,x) in enumerate(existing_train_features) if x is None]

    existing_test_features = [get_from_cache((tf,m,task.get('transform_average')),FEATURE_CACHE) for tf in test_filenames]
    existing_test_labels = [test_labels[i] for (i,x) in enumerate(existing_test_features) if x is not None]
    existing_test_filenames = [test_filenames[i] for (i,x) in enumerate(existing_test_features) if x is not None]
    new_test_filenames = [test_filenames[i] for (i,x) in enumerate(existing_test_features) if x is None]
    reordered_test_filenames = existing_test_filenames + new_test_filenames 
    new_test_labels = [test_labels[i] for (i,x) in enumerate(existing_test_features) if x is None]

    if convolve_func_name == 'numpy':
        num_batches = multiprocessing.cpu_count()
        if num_batches > 1:
            print('found %d processors, using that many processes' % num_batches)
            pool = multiprocessing.Pool(num_batches)
            print('allocated pool')
        else:
            pool = multiprocessing.Pool(1)
    elif convolve_func_name == 'cufft':
        num_batches = get_num_gpus()
        #num_batches = 1
        if num_batches > 1:
            print('found %d gpus, using that many processes' % num_batches)
            pool = multiprocessing.Pool(processes = num_batches)
        else:
            pool = multiprocessing.Pool(1)
    else:
        raise ValueError, 'convolve func name not recognized'

    print('num_batches',num_batches)
    if num_batches > 0:
        batches = get_data_batches(new_train_filenames,num_batches)
        results = []
        for (bn,b) in enumerate(batches):
            results.append(pool.apply_async(extract_inner_core,(b,m.to_dict(),convolve_func_name,bn,task.to_dict(),cache_port)))
        results = [r.get() for r in results]
        new_train_features = ListUnion(results)
        batches = get_data_batches(new_test_filenames,num_batches)
        results = []
        for (bn,b) in enumerate(batches):
            results.append(pool.apply_async(extract_inner_core,(b,m.to_dict(),convolve_func_name,bn,task.to_dict(),cache_port)))
        results = [r.get() for r in results]
        new_test_features = ListUnion(results)
    else:
        print('train feature extraction ...')
        new_train_features = extract_inner_core(new_train_filenames,m,convolve_func_name,0,task,cache_port)
        print('test feature extraction ...')
        new_test_features = extract_inner_core(new_test_filenames,m,convolve_func_name,0,task,cache_port)


    #TODO get the order consistent with original ordering
    train_features = filter(lambda x : x is not None,existing_train_features) + new_train_features
    train_features,train_feature_info = zip(*train_features)
    train_features = sp.row_stack(train_features)
    test_features = filter(lambda x : x is not None, existing_test_features) + new_test_features
    test_features,test_feature_info = zip(*test_feature)
    test_features = sp.row_stack(test_features)
    train_labels = existing_train_labels + new_train_labels
    test_labels = existing_test_labels + new_test_labels
    
    for (im,f) in zip(new_train_filenames,new_train_features):
        put_in_cache((im,m,task.get('transform_average')),f,FEATURE_CACHE)
    for(im,f) in zip(new_test_filenames,new_test_features):
        put_in_cache((im,m,task.get('transform_average')),f,FEATURE_CACHE)
          
    res = train_test(task,reordered_train_filenames,reordered_test_filenames,train_labels,test_labels,train_features,test_features,classifier_kwargs)
    res['feature_info'] = get_feature_info_summary(train_features,test_features,train_feature_info,test_feature_info)
    return res

def put_in_performance(split_results,image_config_gen,m,model_hash,image_hash,perf_coll,task,ext_hash,extraction=None,extraction_hash=None):
    
    model_results = SON([])
    for stat in STATS:
        if stat in split_results[0] and split_results[0][stat] != None:
            model_results[stat] = sp.array([split_result[stat] for split_result in split_results]).mean()        
            
    feature_info = summarize_feature_info_over_splits([split_result['feature_info'] for split_result in split_results])

    out_record = SON([('model',m['config']['model']),
                      ('model_hash',model_hash), 
                      ('model_filename',m['filename']), 
                      ('images',son_escape(image_config_gen['images'])),
                      ('image_hash',image_hash),
                      ('task',son_escape(task)),
                      ('feature_info',feature_info),
                      ('__hash__',ext_hash)
                 ])
                 
    if extraction:
        out_record['extraction'] = son_escape(extraction)
        out_record['extraction_hash'] = extraction_hash
                                  
    out_record.update(model_results)

    print('inserting result ...')
    perf_coll.insert(out_record)
    
    return model_results


def put_in_split(split,image_config_gen,m,task,ext_hash,split_id,split_fs,extraction=None):
    out_record = SON([('model',m['config']['model']),
                      ('images',son_escape(image_config_gen['images'])),
                      ('task',son_escape(task)),
                      ('split_id',split_id),
                 ])   
    if extraction:
        out_record['extraction'] = son_escape(extraction)

    
    filename = get_filename(out_record)
    out_record['filename'] = filename
    out_record['__hash__'] = ext_hash
    print('pickling split ...')
    out_data = cPickle.dumps(SON([('split',split)]))
    print('dump out split ...')
    split_fs.put(out_data,**out_record)

     
def put_in_split_result(res,image_config_gen,m,task,ext_hash,split_id,splitres_fs,extraction=None):
    
    out_record = SON([('model',m['config']['model']),
                      ('images',son_escape(image_config_gen['images'])),
                      ('task',son_escape(task)),
                      ('split_id',split_id),
                 ])   
                 
    if extraction:
        out_record['extraction'] = son_escape(extraction)
            
    filename = get_filename(out_record)
                 
    split_result = SON([])
    for stat in STATS:
        if stat in res and res[stat] != None:
            split_result[stat] = res[stat] 

    out_record['filename'] = filename
    out_record['feature_info'] = res['feature_info']
    out_record['__hash__'] = ext_hash
    out_record.update(split_result)
    print('pickling split result...')
    out_data = cPickle.dumps(SON([('split_result',res)]))
    print('dumping out split result ...')
    splitres_fs.put(out_data,**out_record)          


def get_feature_info_summary(train_features,test_features,train_feature_info,test_feature_info):
    return SON([('feature_length',len(train_features[0])),
                ('num_channels',train_feature_info[0]['c']),
                ('shape',train_feature_info[0]['s'])])
    
def summarize_feature_info_over_splits(infos):
    return infos[0]
    


STATS = ['train_accuracy','test_accuracy','train_ap','test_ap','train_auc','test_auc']  
    

def train_test(task,train_filenames,test_filenames,train_labels,test_labels,train_features,test_features,classifier_kwargs):

    if task.get('target_map'):
        image_col = db['images.files']
        image_fs = gridfs.GridFS(db,'images')
        train_target_maps = get_target_maps(task,train_filenames,image_fs,image_col)
        test_target_maps = get_target_maps(task,test_filenames,image_fs,image_col)
        train_features, train_labels, train_alignment = align_features_and_labels_with_maps(train_target_maps,train_features,train_labels)
        test_features, test_labels, test_alignment = align_features_and_labels_with_maps(test_target_maps,test_features,test_labels)
        
    print('classifier ...')
    res = svm.multi_classify(train_features,train_labels,test_features,test_labels,classifier_kwargs,relabel = task.get('relabel',True))
    print('Split test accuracy', res['test_accuracy'])
    if task.get('target_map'):
        res['cls_data']['train_labels'] = realign_labels(res['cls_data']['train_labels'], train_alignment)
        res['cls_data']['test_labels'] = realign_labels(res['cls_data']['test_labels'], test_alignment)
        res['cls_data']['train_prediction'] = realign_labels(res['cls_data']['train_prediction'], train_alignment)
        res['cls_data']['test_prediction'] = realign_labels(res['cls_data']['test_prediction'], test_alignment)
        if task['target_map'].get('statfunc'):
            pass # what?
    res['cls_data']['train_filenames'] = train_filenames
    res['cls_data']['test_filenames'] = test_filenames

    return res
    
def get_target_maps(task,filenames,image_fs,image_col):
    target_map_config = task['target_map']
    fn_path = target_map_config['mapping_function']
    fn_name = fn_path.split('.')[-1]
    mod_path = '.'.join(fn_path.split('.')[:-1])
    modl = __import__(mod_path,globals(),locals(),[fn_path])
    fn = getattr(modl,fn_name)
    return [fn(image_fs.get_version(filename),image_col.find_one({'filename':filename})) for filename in filenames]
    
def align_features_and_labels_with_maps(maps,features,labels):
    new_feature_list = []
    new_label_list = []
    alignment = []
    count = 0
    for (map,feature,label) in zip(maps,features,labels):
        if feature.ndim == 1:
            feature = np.tile(feature,map.shape[:2] + (1,))
        else:
            feature = np.resize(feature,map.shape[:2] + feature.shape[2:])
        
        ms = map.shape[0]*map.shape[1]
        
        new_features = feature.reshape((ms,) + feature.shape[2:])
        new_feature_list.append(new_features)
        
        label_array = np.column_stack([map.reshape((ms,) + map.shape[2:]) , ms*[label]])
        new_label_list.append(label_array)
                
        aln = ((map.shape[0],map.shape[1]),count,count+ms)
        count += ms
        alignment.append(aln)
    
    new_labels = np.row_stack(new_label_list)
    new_features = np.row_stack(new_feature_list)
    
    return new_features,new_labels,alignment
    
def realign_labels(labels,alignment):
    new_labels = []
    for ((s0,s1),i0,i1) in alignment:
        l = alignment[i0:i1]
        l = l.reshape((s0,s1) + l.shape[1:])
        new_labels.append(l)
    return new_labels
    


def extract_inner_core(images,m,convolve_func_name,device_id,task,cache_port):

    poller = None

    if convolve_func_name == 'cufft':
        convolve_func = cuFFT.LFBCorrCuFFT(device_id=device_id, use_cache=True)
        context = convolve_func.context
    else:
        convolve_func = c_numpy_mixed
        

    conn = pm.Connection(document_class=bson.SON)
    db = conn[DB_NAME]

    perf_coll = db['performance.files']

    model_fs = gridfs.GridFS(db,'models')
    image_fs = gridfs.GridFS(db,'images')

    filter_fh = model_fs.get_version(m['filename'])
    filter = cPickle.loads(filter_fh.read())
    
    L = [get_features(im, image_fs, filter, m, convolve_func,task,poller) for im in images]
    
    if convolve_func_name == 'cufft':
        context.pop()
        
    return L

def get_features(im,im_fs,filter,m,convolve_func,task,network_cache):

    if network_cache and network_cache.sockets:
        sock = network_cache.sockets.keys()[0]
        obj = (im,m,task.get('transform_average')) 
        hash = hashlib.sha1(repr(obj)).hexdigest()
        sock.send_pyobj({'get':hash})
        poll = network_cache.poll(timeout=NETWORK_CACHE_TIMEOUT)
        if poll != []:
            val = sock.recv_pyobj()
        else:
            val = None
            network_cache.unregister(sock)
        if val is not None:
            output = val
        else:
            output = transform_average(compute_features(im, im_fs, filter, m, convolve_func) , task.get('transform_average'),m)
            sock.send_pyobj({'put':(hash,output)})
            sock.recv_pyobj()
    else:
        output = transform_average(compute_features(im, im_fs, filter, m, convolve_func) , task.get('transform_average'),m)
    return output
    


############DATA SPLITTING#############
############DATA SPLITTING#############
############DATA SPLITTING#############
############DATA SPLITTING#############
############DATA SPLITTING#############

def generate_random_sample(task_config,hash,colname):

    ntrain = task_config['sample_size']
    ntest = 0
    N = 1

    query = task_config.get('query',{})
    cqueries = [reach_in('config',query)]
    
    return traintest.generate_multi_split2(DB_NAME,colname,cqueries,N,ntrain,ntest,universe={'__hash__':hash})[0]


def get_extraction_batches(image_hash,task,batch_size):
    if batch_size:
        conn = pm.Connection(document_class=bson.SON)
        db = conn[DB_NAME]
        coll = db['images.files']
        q = reach_in('config',task.get('query',SON([])))
        q['__hash__'] = image_hash
        count = coll.find(q).count()
        num_batches = int(math.ceil(count/float(batch_size)))
        return [(batch_size*ind,batch_size*(ind+1)) for ind in range(num_batches)]
    else:
        return [None]
        
    
def get_extraction_configs(image_hash,task,batch):
    conn = pm.Connection(document_class=bson.SON)
    db = conn[DB_NAME]
    coll = db['images.files']
    q = reach_in('config',task.get('query',SON([])))
    q['__hash__'] = image_hash
    if batch:
        skip = batch[0]
        delta = batch[1] - skip 
        return coll.find(q,fields=['filename','config.image']).skip(skip).limit(delta)
    else:
        return coll.find(q,fields=['filename','config.image'])
        
def generate_splits(task_config,hash,colname,reachin=True):
    base_query = SON([('__hash__',hash)])
    ntrain = task_config.get('ntrain')
    ntest = task_config.get('ntest')
    N = task_config.get('N')
    kfold = task_config.get('kfold')
    balance = task_config.get('balance')
    overlap = task_config.get('overlap')
    univ = task_config.get('universe',SON([]))
    base_query.update(reach_in('config',univ) if reachin else univ)

    query = copy.deepcopy(task_config['query'])
    if isinstance(query,list):
        cqueries = [reach_in('config',q) if reachin else copy.deepcopy(q) for q in query]
        splits = traintest.generate_multi_split2(DB_NAME,colname,cqueries,N,ntrain,
                                               ntest,universe=base_query,
                                               overlap=overlap, balance=balance, kfold=kfold)
    else:
        ntrain_pos = task_config.get('ntrain_pos')
        ntest_pos = task_config.get('ntest_pos')
        cquery = reach_in('config',query) if reachin else copy.deepcopy(query)
        splits = traintest.generate_splits(DB_NAME,colname,cquery,N,ntrain,ntest,
                                           ntrain_pos=ntrain_pos,ntest_pos = ntest_pos,
                                           universe=base_query,use_negate = True,
                                           overlap=overlap)
                                         
    if task.get('query_labels'):
        L = task['query_labels']
        if isinstace(L,str):
            L = ['NOT ' + L, L]
        splits['train_labels'] = [L[t] for t in splits['train_labels']]
        splits['test_labels'] = [L[t] for t in splits['test_labels']]

    return splits


############TRANFORM AVERAGE#############
############TRANFORM AVERAGE#############
############TRANFORM AVERAGE#############
############TRANFORM AVERAGE#############
############TRANFORM AVERAGE#############


def get_info(l):
    num_channels = len(l.keys())
    sh = l[0].shape
    return SON([('c',num_channels),('s',sh)])

def transform_average(input,config,model_config):
    if isinstance(input,list):
        M = model_config['config']['model']
        assert isinstance(M,list) and len(M) == len(input)
        if isinstance(config,list):
            assert len(config) == len(M)
        else:
            config = [copy.deepcopy(config) for ind in range(len(M))]
        args = zip(input,config,[{'config':{'model':m}} for m in M])
        results = [transform_average(inp,conf,m) for (inp,conf,m) in args]
        vec = sp.concatenate([res[0] for res in results])
        info = [res[1] for res in results]
    else:
        vecs = []
        info = []
        for level_input in input.values():
            K = level_input.keys()
            K.sort()
            vec = []
            info.append(get_info(level_input))
            if config:
                for cidx in K:
                    vec.append(average_transform(level_input[cidx],config,model_config))
            else:
                for cidx in K:
                    vec.append(unravel(level_input[cidx]))
            if vec[0].ndim > 0:
                vec = sp.concatenate(vec)
            vecs.append(vec)
        vec = sp.concatenate(vecs)
    
    return vec,info

def average_transform(input,config,M):
    if config['transform_name'] == 'translation':
        if config.get('max',False):
            V = [input.max(1).max(0)]
        elif config.get('percentile'):
            pcts = config['percentile']
            V = [percentile2d(input,pct) for pct in pcts]
        elif config.get('various_stats',False):
            V = [max2d(input),min2d(input),mean2d(input),argmax2d(input),argmin2d(input)]
        else:
            V = [input.sum(1).sum(0)]
        if config.get('fourier',False):
            V = V + [np.abs(np.fft.fft(v)) for v in V]
        if config.get('norm'):
            if config['norm'] == 'max':
                V = V + [v/v.max() for v in V]
            elif config['norm'] == 'sum':
                V = V + [v/v.sum() for v in V]
            else:
                raise ValueError,'norm not recognized'
        return sp.concatenate(V)
            
    elif config['transform_name'] == 'translation_and_orientation':
        model_config = M['config']['model'] 
        assert model_config.get('filter') and model_config['filter']['model_name'] == 'gridded_gabor'
        H = input.sum(1).sum(0) 
        norients = model_config['filter']['norients']
        phases = model_config['filter']['phases']
        nphases = len(phases)
        divfreqs = model_config['filter']['divfreqs']
        nfreqs = len(divfreqs)
        
        output = np.zeros((H.shape[0]/norients,)) 
        
        for freq_num in range(nfreqs):
            for phase_num in range(nphases):
                for orient_num in range(norients):
                    output[nphases*freq_num + phase_num] += H[norients*nphases*freq_num + nphases*orient_num + phase_num]
        
        return output
    elif config['transform_name'] == 'sum_up':
        model_config = M['config']['model']
        Lo = model_config['layers'][-2]
        L = model_config['layers'][-1]
        s1 = len(Lo['filter']['divfreqs'])
        s2 = Lo['filter']['norients']
        if L['filter']['model_name'] == 'uniform':
            os = L['filter'].get('osample',1)
            fs = L['filter'].get('fsample',1)
            s1 = s1/fs
            s2 = s2/os
        elif L['filter']['model_name'] == 'freq_uniform':
            os = L['filter'].get('osample',1)
            s2 = s2/os            
        return sum_up(input.max(1).max(0),s1,s2)
    elif config['transform_name'] == 'nothing':
        return input.ravel()
    elif config['transform_name'] == 'translation_and_fourier':
        return np.abs(np.fft.fft(input.sum(1).sum(0)))
    elif config['transform_name'] == 'central_slice':
        return get_central_slice(input,config['ker_shape'])
    else:
        raise ValueError, 'Transform ' + str(config['transform_name']) + ' not recognized.'

def feature_postprocess(vec,config,m,extraction):
    if config:
        if config['transform_name'] == 'subranges':
            assert extraction['transform_average']['transform_name'] == 'translation'
            num_layers = len(m['config']['model']['layers'])
            filter_sizes = model_generation.get_filter_sizes(m['config']['model']['layers'])
            filter_sizes = dict(zip(range(num_layers),filter_sizes))
            filter_sizes[-1] = 1
            num_percts = len(extraction['transform_average'].get('percentile',[100]))
            layers = config.get('layers')
            if layers is None: layers = range(-1,num_layers)
            percts = config.get('percts')
            if percts is None: percts = range(num_percts)

            feedup = m['config']['model'].get('feed_up',False)
            if feedup:
                assert len(vec) == sum([f*num_percts for f in filter_sizes.values()])            
            else:
                assert len(vec) == filter_sizes[num_layers-1]*num_percts

            vec1 = vec[0:0]
            for l in layers:
                if feedup:
                    ind0 = sum([num_percts*filter_sizes[f0] for f0 in filter_sizes if f0 < l])
                else:
                    ind0 = 0
                f = filter_sizes[l]
                for p in percts:
                    vec1 = np.append(vec1,vec[ind0+f*p:ind0+f*(p+1)])
            
            assert len(vec1) == sum([filter_sizes[l]*len(percts) for l in layers])
             
            return vec1
      

        else:
            raise ValueError, 'Transform ' + str(config['transform_name']) + ' not recognized.'
    else:
        return vec



############CORE COMPUTATIONS###############
############CORE COMPUTATIONS###############
############CORE COMPUTATIONS###############
############CORE COMPUTATIONS###############
############CORE COMPUTATIONS###############
    
    
def compute_features(image_filename, image_fs, filter, model_config, convolve_func):
    image_fh = image_fs.get_version(image_filename)
    print('extracting', image_filename, model_config)
    return compute_features_core(image_fh,filter,model_config,convolve_func)    


def compute_features_core(image_fh,filters,model_config,convolve_func):
 
    m_config = model_config['config']['model']
    
    if isinstance(m_config,list):
        reslist = []
        for (filt,m) in zip(filters,m_config):
            image_fh.seek(0)
            res = compute_features_core(image_fh,filt,{'config':{'model':m}},convolve_func)
            reslist.append(res)
        return reslist
    else:
        conv_mode = m_config['conv_mode']    
        layers = m_config['layers']
        feed_up = m_config.get('feed_up',False)
        
        array = image2array(m_config,image_fh)
        array,orig_imga = preprocess(array,m_config)
        assert len(filters) == len(layers)
        dtype = array[0].dtype
        
        array_dict = {}
        for (ind,(filter,layer)) in enumerate(zip(filters,layers)):
 
            if feed_up:
                array_dict[ind-1] = array       

            if filter is not None:
                filter = fix_filter(array[0],filter)

            if isinstance(layer,list):
                arrays = [compute_layer(array,filter,l,convolve_func,conv_mode) for l in layer]
                array = harmonize_arrays(arrays,model_config)
            else:
                array = compute_layer(array,filter,layer,convolve_func,conv_mode)

        array_dict[len(layers)-1] = array
            
        return array_dict


def fix_filter(array,filter):
    if array.ndim == 3:
        if filter.ndim == 3:
            filter = filter.reshape(filter.shape + (1,))
        if filter.shape[3] < array.shape[2]:
            assert array.shape[2] % filter.shape[3] == 0
            reps = array.shape[2]/filter.shape[3]
            filter = np.repeat(filter,reps,axis=3)

    return filter

def compute_layer(array,filter,layer,convolve_func,conv_mode):  
    if layer.get('scales') is not None:
        scales = layer['scales']
        sh = array[0].shape[:2]
        get_sh = lambda x : (int(round(sh[0]*x)),int(round(sh[1]*x)))
        arrays = [compute_inner_layer(resample(array,get_sh(scale),layer),filter,layer,convolve_func,conv_mode) for scale in scales]
        return harmonize_arrays(arrays,layer)
    else:
        return compute_inner_layer(array,filter,layer,convolve_func,conv_mode)

def compute_inner_layer(array,filter,layer,convolve_func,conv_mode):
    print('filter',array[0].shape)
    if filter is not None:
        array = fbcorr(array, filter, layer , convolve_func)

    print('lpool',array[0].shape)
    if layer.get('lpool'):
        array = lpool(array,conv_mode,layer['lpool'])

        print('lnorm',array[0].shape)
    if layer.get('lnorm'):
        if layer['lnorm'].get('use_old',False):
            array = old_norm(array,conv_mode,layer['lnorm'])
        else:
            array = lnorm(array,conv_mode,layer['lnorm'])
            
    return array
    
import numpy as np

def fix_1ds(array):
    adict = {}
    for k in array:
        if array[k].ndim > 2:
            return array
        else:
            adict[k] = array[k].reshape(array[k].shape + (1,))
    return adict
    
def resample(array,scale,config):
    adict = {}
    for k in array:
        sh = array[k].shape
        new_sh = scale + sh[2:]
        adict[k] = imresize(array[k],new_sh,mode='nearest')
    return adict
  
from scipy.ndimage.interpolation import affine_transform
def imresize(x,sh,**kwargs):
    m = np.array(x.shape)/np.array(sh).astype(float)
    print(m)
    return affine_transform(x,m,output_shape = sh,**kwargs)
        
def harmonize_arrays(arrays,config):
    arrays = [fix_1ds(array) for array in arrays]
    sizes = [array[0].shape for array in arrays]
    max0 = max([s[0] for s in sizes])
    max1 = max([s[1] for s in sizes])
    new_sh = (max0,max1)
    keylist = [array.keys() for array in arrays]
    assert all([keylist[0] == kk for kk in keylist])
    arrays = [resample(array,new_sh,config) for array in arrays]
    return dict_concatenate(arrays)

def dict_concatenate(arrays):
    adict = {}
    for k in arrays[0]:
        adict[k] = np.concatenate([array[k] for array in arrays],axis=2)
    return adict



def fbcorr(input,filter,layer_config,convolve_func):
    output = {}     
    for cidx in input.keys():
        if layer_config['filter']['model_name'] == 'multiply':
            (s1,s2) = filter
            output[cidx] = multiply(input[cidx],s1,s2,all=layer_config['filter'].get('all',False),
                                    max=layer_config['filter'].get('max',False),
                                    ravel=layer_config['filter'].get('ravel',False))
        else:
            min_out = layer_config['activ'].get('min_out')
            max_out=layer_config['activ'].get('max_out')
            if hasattr(min_out,'__iter__') and not hasattr(max_out,'__iter__'):
                max_out = [max_out]*len(min_out)
            elif hasattr(max_out,'__iter__') and not hasattr(min_out,'__iter__'):
                min_out = [min_out]*len(max_out)
            if hasattr(min_out,'__iter__'):
                output[cidx] = convolve_func(input[cidx],
                                             filter,
                                             mode=layer_config['filter'].get('mode','valid'))                
                for ind  in range(output[cidx].shape[2]):
                    output[cidx][:,:,ind] = output[cidx][:,:,ind].clip(min_out[ind],max_out[ind])
            else:
                output[cidx] = convolve_func(input[cidx],
                                             filter,
                                             min_out=min_out,
                                             max_out=max_out,
                                             mode=layer_config['filter'].get('mode','valid'))         
    return output


def old_norm(input,conv_mode,params):
    output = {}
    for cidx in input.keys():
        if len(input[cidx].shape) == 3:
            inobj = input[cidx]
            strip = False
        else:
            strip = True
            inobj = input[cidx][:,:,sp.newaxis]

        if params:
    
            res = v1f.v1like_norm(inobj, conv_mode, params['inker_shape'],params['threshold'])
            if strip:
                res = res[:,:, 0]
            output[cidx] = res
        else: 
            output[cidx] = inobj
            
    return output


def lpool(input,conv_mode,config):
    if isinstance(config,list):
        pooled_list = [lpool(input,conv_mode,c) for c in config]
        pooled = harmonize_arrays(pooled_list)
    else:
        pooled = {}
        if hasattr(config.get('order'),'__iter__') or config.get('percentile') or config.get('rescale'):
            plugin = 'numpy_naive'
        else:
            plugin = 'cthor'
            
        for cidx in input.keys():
            pooled[cidx] = pythor3.operation.lpool(input[cidx],plugin=plugin,**config) 

    return pooled

def lnorm(input,conv_mode,config):
    if isinstance(config,list):
        normed_list = [lnorm(input,conv_mode,c) for c in config]
        normed = harmonize_arrays(normed_list)
    else:
        normed = {}
        if 'inker_shape' in config: 
            config['inker_shape'] = tuple(config['inker_shape'])
        if 'outker_shape' in config:
            config['outker_shape'] = tuple(config['outker_shape'])
        if hasattr(config.get('threshold'),'__iter__') or hasattr(config.get('stretch'),'__iter__'):
            plugin = 'numpy_naive'
        else:
            plugin = 'cthor'    
        for cidx in input.keys():
            normed[cidx] = pythor3.operation.lnorm(input[cidx],plugin=plugin,**config)    
    
    return normed

def c_numpy_mixed(arr_in, arr_fb, arr_out=None,
                 mode=DEFAULT_MODE,
                 min_out=DEFAULT_MIN_OUT,
                 max_out=DEFAULT_MAX_OUT,
                 stride=DEFAULT_STRIDE
                ):
    
    if max(arr_fb.shape[-3:-1]) > 19:
        return pythor3.operation.fbcorr(arr_in, arr_fb, arr_out=None,
                 mode=mode,
                 min_out=min_out,
                 max_out=max_out,
                 stride=stride,
                 plugin="numpyFFT",
                 plugin_kwargs={"use_cache":True})
    else:
        return pythor3.operation.fbcorr(arr_in, arr_fb, arr_out=None,
                 mode=mode,
                 min_out=min_out,
                 max_out=max_out,
                 stride=stride,
                 plugin="cthor")



###########UTILS#########
###########UTILS#########
###########UTILS#########
###########UTILS#########
###########UTILS#########


def get_from_cache(obj,cache):
    hash = hashlib.sha1(repr(obj)).hexdigest()
    if hash in cache:
        print('using cache for %s' % str(hash))
        return cache[hash]
        
def put_in_cache(obj,value,cache):
    hash = hashlib.sha1(repr(obj)).hexdigest()
    cache[hash] = value
     

def get_num_gpus():
    p = multiprocessing.Pool(1)
    r = p.apply(get_num_gpus_core,())
    return r

def get_num_gpus_core():
    cuda.init()
    num = 0
    while True:
        try:
            cuda.Device(num)
        except:
            break
        else:
            num +=1
    return num


def get_data_batches(data,num_batches):

    bs = int(math.ceil(float(len(data)) / num_batches))
    
    return [data[bs*i:bs*(i+1)] for i in range(num_batches)]

    
def unravel(X):
    return sp.concatenate([X[:,:,i].ravel() for i in range(X.shape[2])])
        
def sum_up(x,s1,s2):
    y = []
    F = lambda x : (s2**2)*x[2] + s2*x[0] + x[1]
    for i in range(s1):
        S = [map(F,[(k,(j+k)%s2,i) for k in range(s2)]) for j in range(s2)]
        y.extend([sum([x[ss] for ss in s]) for s in S])
    return np.array(y)
    
def mean2d(x):
    if x.ndim <= 2:
        return np.array([x.mean()])
    else:
        return np.array([x[:,:,i].mean() for i in range(x.shape[2])])

def max2d(x):
    if x.ndim <= 2:
        return np.array([x.max()])
    else:
        return np.array([x[:,:,i].max() for i in range(x.shape[2])])
    
def min2d(x):
    if x.ndim <= 2:
        return np.array([x.min()])
    else:
        return np.array([x[:,:,i].min() for i in range(x.shape[2])])

def argmax2d(x):
    if x.ndim <= 2:
        return np.array([x.argmax()])
    else:
        return np.array([x[:,:,i].argmax() for i in range(x.shape[2])])

def argmin2d(x):
    if x.ndim <= 2:
        return np.array([x.argmin()])
    else:
        return np.array([x[:,:,i].argmin() for i in range(x.shape[2])])

def percentile2d(x,pct):
    if x.ndim <= 2:
        return np.array([stats.scoreatpercentile(x.ravel(),pct)])
    else:
        return np.array([stats.scoreatpercentile(x[:,:,i].ravel(),pct) for i in range(x.shape[2])])





