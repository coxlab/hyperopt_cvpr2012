from starflow.utils import creates, activate, is_string_like, ListUnion, uniqify, get_argd, is_string_like

#from collections import OrderedDict
import itertools
import gridfs
import hashlib
import os
import datetime
import time
import random
import cPickle
from starflow.utils import activate
import pymongo as pm
from bson import SON,BSON
import numpy as np



#############general DB things

CERTIFICATE_ROOT = '../.db_certificates'

def initialize_certificates(creates = CERTIFICATE_ROOT):
    MakeDir(creates)


#####high-level functions

def do_initialization(initialize,args):
    oplist = initialize(*args)   
    db_ops_initialize(oplist)
    return oplist


def DBAdd(initialize,args = ()):
    """
        main DB protocol generator.   this (and the decorators) are the main
        thing used by the user
    """
    oplist = do_initialization(initialize,args)  

    D = [(a['step'],a['func'].meta_action,(a['func'],a.get('incertpaths'),a['outcertpaths'],a.get('params'))) for a in oplist]  
    return D
    

def inject(dbname,outroots,generator,setup=None,cleanup=None,caches=None):
    if is_string_like(outroots):
        outroots = [outroots]
    def func(f):
        f.meta_action = inject_op
        f.action_name = 'inject'
        f.dbname = dbname
        f.inroots = ['']
        f.outroots = outroots
        f.generator = generator
        f.setup = setup
        f.cleanup = cleanup

        return f
        
    return func
    
    
def dot(dbname,inroots,outroots,setup=None,cleanup=None):
    if is_string_like(inroots):
        inroots = [inroots]
    if is_string_like(outroots):
        outroots = [outroots]
    def func(f):
        f.dbname = dbname
        f.meta_action = dot_op
        f.action_name = 'dot'
        f.inroots = inroots
        f.outroots = outroots
        f.setup = setup
        f.cleanup = cleanup

        return f
    return func
    
    
def cross(dbname,inroots,outroots,setup=None,cleanup=None):
    if is_string_like(inroots):
        inroots = [inroots]
    if is_string_like(outroots):
        outroots = [outroots]
    def func(f):
        f.dbname = dbname 
        f.meta_action = cross_op
        f.action_name = 'cross'
        f.inroots = inroots
        f.outroots = outroots
        f.setup = setup
        f.cleanup = cleanup
        return f
        
    return func
    

def aggregate(dbname,inroots,aggregate_on,outroots,setup=None,cleanup=None):   
    if is_string_like(inroots):
        inroots = [inroots]
    if is_string_like(outroots):
        outroots = [outroots]
    def func(f):
        f.dbname = dbname 
        f.aggregate_on = aggregate_on
        f.meta_action = aggregate_op
        f.action_name = 'aggregate'
        f.inroots = inroots
        f.outroots = outroots
        f.setup = setup
        f.cleanup = cleanup
        return f
        
    return func    
    
    
########main db operations
   
def op_depends(x):
    """
        generates paths of read certificates
    """
    return tuple(x[1])


def op_creates(x):
    """
        generates paths of write certificates
    """
    return tuple(x[2])


@activate(lambda x : (), op_creates)
def inject_op(func,incertpaths,outcertpaths,params):
    """
       use "func" to inject new data into a source data collection
    """
    configs = func.generator(*params)
    
    assert incertpaths is None
    incertdicts = load_incerts(func,None,outcertpaths,params)
            
    outroots = func.outroots
    dbname = func.dbname
    conn = pm.Connection(document_class=SON)
    db = conn[dbname]
    out_fs = [gridfs.GridFS(db,collection = coll) for coll in outroots]
    out_cols = [db[coll + '.files'] for coll in outroots]
    
    if func.setup:
        pass_args = func.setup() or {}
    else:
        pass_args = {}    
    
    config_time = FuncTime(func)

    for config in configs:
        assert isinstance(config,SON)
        if not already_exists(config,out_cols,config_time,func):
            results = do_rec(None,  config, func, pass_args)
            for (fs,res) in zip(out_fs,results):
                outdata,res = interpret_res(res,config,func)
                fs.put(res,**outdata)
                
    if func.cleanup:
        func.cleanup()   
        
    write_outcerts(func,configs,incertdicts,outcertpaths,db)

        
@activate(op_depends,op_creates)
def dot_op(func,incertpaths,outcertpaths,params):
    """
        takes "zip" of source collection parameters in computing output collections
    """

    inroots = func.inroots
    outroots = func.outroots
    dbname = func.dbname
    conn = pm.Connection(document_class=SON)
    db = conn[dbname]
    in_fs = [gridfs.GridFS(db,collection = coll) for coll in inroots]
    in_cols = [db[coll + '.files'] for coll in inroots]
    out_cols = [db[coll + '.files'] for coll in outroots]
    out_fs = [gridfs.GridFS(db,collection = coll) for coll in outroots]
    
    ftime = FuncTime(func)
    if func.setup:
        pass_args = func.setup() or {}
    else:
        pass_args = {}

    if params is None:
        params = SON([])
    
    incertdicts = load_incerts(func,incertpaths,outcertpaths,params)

    data_list = get_data_list(in_cols,func)
    data_list = zip(*data_list)

    newdata_list = []
    for data_tuple in data_list:         
        filenames = [dt['filename'] for dt in data_tuple]
        data_tuple = [dt['config'] for dt in data_tuple]
        fhs = [fs.get_version(filename) for (fs,filename) in zip(in_fs,filenames)]
        config_time = max(ftime,*[get_time(fh.upload_date) for fh in fhs])
        
        newdata = dict_union(data_tuple)
        newdata.update(params)
        newdata_list.append(newdata)
 
        if not already_exists(newdata,out_cols,config_time,func):
            results =  do_rec(fhs, newdata, func, pass_args) 
            for (fs,res) in zip(out_fs,results):
                outdata,res = interpret_res(res,newdata,func)
                fs.put(res,**outdata)
                 
    if func.cleanup:
        func.cleanup()     
        
    write_outcerts(func,newdata_list,incertdicts,outcertpaths,db)
    

    
@activate(op_depends,op_creates)
def cross_op(func,incertpaths,outcertpaths,params):
    """
        takes "product" of source collection parameters in computing output collections
    """
    
    inroots = func.inroots
    outroots = func.outroots
    dbname = func.dbname
    conn = pm.Connection(document_class=SON)
    db = conn[dbname]
    in_cols = [db[coll + '.files'] for coll in inroots]
    out_cols = [db[coll + '.files'] for coll in outroots]
    in_fs = [gridfs.GridFS(db,collection = coll) for coll in inroots]
    out_fs = [gridfs.GridFS(db,collection = coll) for coll in outroots]    
    ftime = FuncTime(func)      
    if func.setup:
        pass_args = func.setup() or {}
    else:
        pass_args = {}

    if params is None:
        params = SON([])

    incertdicts = load_incerts(func,incertpaths,outcertpaths,params) 
           
    data_list = get_data_list(in_cols,func)
 
    data_product = list(itertools.product(*data_list))
    
    newdata_list = []
    for data_tuple in data_product:
        
        filenames = [dt['filename'] for dt in data_tuple]

        data_tuple = [dt['config'] for dt in data_tuple]
        fhs = [fs.get_version(filename) for (fs,filename) in zip(in_fs,filenames)]
        config_time = max(ftime,*[get_time(fh.upload_date) for fh in fhs])
        
        data_tuple = tuple(list(data_tuple) + [params])
        flat_data = dict_union(data_tuple)        
        newdata_list.append(flat_data)

        if not already_exists(flat_data,out_cols,config_time,func):
            
            results = do_rec(fhs, data_tuple, func, pass_args)
            for (fs,res) in zip(out_fs,results):
                outdata,res = interpret_res(res,flat_data,func)
                fs.put(res,**outdata)

     
    if func.cleanup:
        func.cleanup()  
        
    write_outcerts(func,newdata_list,incertdicts,outcertpaths,db)


@activate(op_depends,op_creates)
def aggregate_op(func,incertpaths,outcertpaths,params):

    aggregate_on = func.aggregate_on
    
    inroots = func.inroots
    outroots = func.outroots
    dbname = func.dbname
    conn = pm.Connection(document_class=SON)
    db = conn[dbname]
    in_cols = [db[coll + '.files'] for coll in inroots]
    out_cols = [db[coll + '.files'] for coll in outroots]
    in_fs = [gridfs.GridFS(db,collection = coll) for coll in inroots]
    out_fs = [gridfs.GridFS(db,collection = coll) for coll in outroots]    

    ftime = FuncTime(func)      
    if func.setup:
        pass_args = func.setup() or {}
    else:
        pass_args = {}
  

    incertdicts = load_incerts(func,incertpaths,outcerpaths,params)    
    
    data_list = get_data_list(in_cols,func)
        
    params = func.params
    if params is None:
        params = SON([])   
    
    ind = func.inroots.index(aggregate_on)
    aggregate_params = incertdicts[ind]['param_names'][aggregate_on]
    aggregate_val = son_escape(func.in_args[ind])
    
    D = get_aggregate(data_list,aggregate_val,aggregate_params,aggregate_on,params)

    for (dtuples,Ndict) in D.values():
        filenames = [[data['filename'] for data in data_tuple] for data_tuple in dtuples]
        filehandles = [[fs.get_version(fname) for (fs,fname) in zip(in_fs,fnames)] for fnames in filenames]
        config_time = max([max(ftime,*[get_time(fh.upload_date) for fh in fhs]) for fhs in filehandles])
        
        dtuples = [[dt['config'] for dt in data_tuple] for data_tuple in dtuples]
        if not already_exists(Ndict,out_cols,config_time,func):
            results =  do_rec(filehandles, dtuples, func, pass_args) 
            for (fs,res) in zip(out_fs,results):
                outdata,res = interpret_res(res,Ndict,func)
                fs.put(res,**outdata)
         
    if func.cleanup:
        func.cleanup()      
        
    newdata_list = [x[1] for x in D.values()]    
    write_outcerts(func,newdata_list,incertdicts,outcertpaths,db)


def get_aggregate(config_list,aggregate_val,aggregate_params,aggregate_on,params):
    config_list = zip(*config_list)
    D = {}
    for config_tuple in config_list:
        N = []
        for c in config_tuple:
            c['config'].update(params) 
            nonagg_params = set(c['config'].keys()).difference(aggregate_params)
            nonagg_values = SON([(p,c['config'][p]) for p in nonagg_params])
            N.append(nonagg_values)
        Ndict = dict_union(N)
        Ndict[aggregate_on + '__aggregate__'] = aggregate_val 
        
        
        r = repr(Ndict)
        if r in D:
            D[r][0].append(config_tuple)
        else:
            D[r] = ([config_tuple],Ndict)
            
    return D


#######technical dependencies

def get_dep(x,att):
    deps = []
    if hasattr(x[0].meta_action,att):
        deps += getattr(x[0].meta_action,att)(x)
    
    if hasattr(x[1],att):
        args = get_argd(x[2]) 
        deps += getattr(x[1],att)(args)
    return tuple(deps)
    
    
def get_op_gen(op,oplist):
    
    if op.get('outcertpaths') is None:
        func = op['func']
        params = op.get('params')
        inroots = func.inroots
        outroots = func.outroots
        if func.action_name == 'inject':
            args = op['params']  
            out_args = SON([(outroot,params) for outroot in outroots])
                 
        else:
            params = op.get('params',SON([])) 
  
            parents = []
            for ir in inroots:
                try:
                    parent = [op0 for op0 in oplist if ir in op0['func'].outroots][0]
                except IndexError:
                    raise IndexError, 'No parent found for at least one collection in ' + repr(op0['func'].outroots) 
                else:
                    parents.append(parent)
  
            for parent in parents:
                get_op_gen(parent,oplist)
                
            in_args = [parent['out_args'] for parent in parents]
            op['incertpaths'] = [get_cert_path(func.dbname,inroot,get_config_string(in_arg)) for (inroot,in_arg) in zip(inroots,in_args)]
            out_args = dict_union(in_args)
            out_args.update(params)
            
       
        op['out_args'] = out_args
        op['outcertpaths'] = [get_cert_path(func.dbname,outroot,get_config_string(out_args)) for outroot in func.outroots]

            

#######utils
def random_id():
    return hashlib.sha1(str(np.random.randint(10,size=(32,)))).hexdigest()    


def check_args(args):
    BSON.encode(args,check_keys=True)

def db_ops_initialize(oplist):
    for op in oplist:
        print('Initializing', op)
        get_op_gen(op,oplist)
        

def load_incerts(func,incertpaths,outcertpaths,params):   
    
    if incertpaths:
        incertdicts =  [cPickle.load(open(incertpath)) for incertpath in incertpaths]   
        in_args = [d['out_args'] for d in incertdicts]
        func.inconfig_strings = [get_config_string(a) for a in in_args]
        assert all([d['db'] == func.dbname and d['root'] == coll and d['config_hash'] == s for (coll,s,d) in zip(func.inroots,func.inconfig_strings,incertdicts)])
        outargs = dict_union(in_args)
        outargs.update(params)
        
        func.inrun_hashes = [d['run_hash'] for d in incertdicts]
        
    else:
        incertdicts = None
        outargs = SON([(outroot,params) for outroot in func.outroots])
    
    func.outrun_hash = random_id()
    func.out_args = outargs
    func.outconfig_string = get_config_string(func.out_args)
    assert outcertpaths == [get_cert_path(func.dbname, root, func.outconfig_string) for root in func.outroots]
    
    return incertdicts

   
def write_outcerts(func,configs,incertdicts,outcertpaths,db):
    if incertdicts:
        old_param_names = dict_union([op['param_names'] for op in incertdicts])
    else:
        old_param_names = SON([])
        
    new_param_names = uniqify(ListUnion([x.keys() for x in configs])) 
    for (outcertpath,outroot) in zip(outcertpaths,func.outroots):
        param_names = old_param_names.copy()
        param_names[outroot] = new_param_names
        remove_incorrect(db,outroot,func.outconfig_string,func.outrun_hash)
        createCertificateDict(outcertpath,{'run_hash':func.outrun_hash,'db':func.dbname, 'out_args': func.out_args, 'root':outroot, 'config_hash':func.outconfig_string, 'param_names':param_names})    
        
        
def remove_incorrect(db,root,config_hash,run_hash):
    coll = db[root + '.files']
    coll.update({'__config_hash__':config_hash,'__run_hash__':{'$ne':run_hash}},{'$pull':{'__config_hash__':config_hash}},multi=True)


def createCertificateDict(path,d,tol=10000000000):
    d['__certificate__'] = random.randint(0,tol)
    dir = os.path.split(path)[0]
    os.makedirs2(dir)
    F = open(path,'w')
    cPickle.dump(d,F)
    F.close()
    
  
def do_rec(in_fhs,config,func,pass_args):
    print('Computing',func.__name__,config)
    if in_fhs:
        results = func(in_fhs,config,**pass_args)          
    else:
        results =  func(config,**pass_args)
    
    if not (isinstance(results,tuple) or isinstance(results,list)):
        results = [results]
        
    return results
    

def dict_union(dictlist):
    newdict = dictlist[0].copy()
    for d in dictlist[1:]:
        newdict.update(d)
        
    return newdict 
    
def get_time(dt):
    #return time.mktime(dt.timetuple()) + dt.microsecond*(10**-6) 
    return dt.timetuple()
    
from starflow import linkmanagement, storage
def FuncTime(func):
    modulename = func.__module__
    funcname = func.__name__
    modulepath = '../' + modulename + '.py'
    fullfuncname = modulename + '.' + funcname
    Seed = [fullfuncname]
    Up = linkmanagement.UpstreamLinks(Seed)
    check = zip(Up['SourceFile'],Up['LinkSource']) + [(modulepath,funcname)]
    checkpaths = Up['SourceFile'].tolist() + [modulepath]
    times = storage.ListFindMtimes(check,depends_on = tuple(checkpaths))
    return time.gmtime(max(times.values()))

def reach_in(attr,q):
    q1 = SON([])
    for k in q:
        if k == '$or':
            q1[k] = [reach_in(attr,l) for l in q[k]]
        elif k.startswith('$'):
            q1[k] = q[k]
        else:
            q1[attr + '.' + k] = q[k]
    return q1   
    

def interpret_res(res,data,func):
    cstr = func.outconfig_string
    run_hash = func.outrun_hash
    datacopy = data.copy()
    
    if not isinstance(res,str):
        assert isinstance(res,dict)
        file_res = res.pop('__file__','')
        datacopy.update(res)
    else:
        file_res = res
        
    outdata = {'config' : datacopy, '__config_hash__' : [cstr],'__run_hash__':[run_hash]}
    outdata['filename'] = get_filename(data) 


    return outdata,file_res
    
    
def already_exists(config,coll_list,t, func):
    cstr = func.outconfig_string
    run_hash = func.outrun_hash
    
    q = reach_in('config',config)
    #d = datetime.datetime.fromtimestamp(t)
    d = datetime.datetime(*t[:6])
    q['uploadDate'] = {'$gte':d}
    recs = [coll.find_one(q) for coll in coll_list]
    
    all_exist = all(recs)
    
    assert (not any(recs)) or all_exist
    
    if all_exist:
        print('Exists',q)
        for (coll,rec) in zip(coll_list,recs):
            coll.update(q,{'$addToSet':{'__config_hash__':cstr},'$addToSet':{'__run_hash__':run_hash}})   #the relevant mongo thing
    
    return all_exist

def get_cert_path(dbname,root,config_string):
    return os.path.join(CERTIFICATE_ROOT,dbname,root,config_string)
    
def get_config_string(configs):
    return hashlib.sha1(repr(configs)).hexdigest()
    
def get_filename(config):
    return hashlib.sha1(repr(config)).hexdigest()    

def get_data_list(in_cols,func): 
    inconfig_strings = func.inconfig_strings
    inrun_hashes = func.inrun_hashes
    return [get_most_recent_files(in_col,{'__config_hash__':cstr,'__run_hash__':run_hash},kwargs={'fields':['config','filename']}) for (in_col,cstr,run_hash) in zip(in_cols,inconfig_strings,inrun_hashes)]

def get_most_recent_files(coll,q,skip=0,limit=0,kwargs=None):
    if kwargs is None:
        kwargs = {}
    c = coll.find(q,**kwargs).sort([("filename", 1), ("uploadDate", -1)]).skip(skip).limit(limit) 
    cl = list(c)
    return get_recent(cl)
        
def get_recent(filespecs):
    return [f for (i,f) in enumerate(filespecs) if i == 0 or filespecs[i-1]['filename'] != f['filename']]
    
def son_key_escape(k):
    return k.replace('$','\$').replace('.','__')

def son_escape(x):
    if isinstance(x,SON):
        return SON([(son_key_escape(k),son_escape(v)) for (k,v) in x.items()])    
    elif isinstance(x,list):
        return [son_escape(y) for y in x]
    elif isinstance(x,tuple):
        return tuple([son_escape(y) for y in x])
    else:
        return x    
        
def hsetattr(d,k,v):
    kl = k.split('.')
    h_do_setattr(d,kl,v)
    
def h_do_setattr(d,kl,v):
    if len(kl) == 1:
        d[kl[0]] = v
    else:
        h_do_setattr(d[kl[0]],kl[1:],v)
        
def hgetattr(d,k):
    kl = k.split('.')
    return h_do_getattr(d,kl)
    
def h_do_getattr(d,kl):
    if len(kl) == 1:
        return d[kl[0]]
    else:
        return h_do_getattr(d[kl[0]],kl[1:]) 
        