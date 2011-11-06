import cPickle
import itertools
import math  
import pymongo as pm
import scipy as sp
from bson import SON
import tabular as tb
import gridfs
from starflow.utils import uniqify, ListUnion

from dbutils import get_most_recent_files, dict_union

import numpy as np
import tabular as tb    
import copy



"""
train / test 
"""

def combine_things(a,b):

    for k in b:
        if k == '$where' and k in a:
            a[k] = a[k].strip('; ') + ' && ' + b[k]                
        elif k == '$or' and k in a:
            pass
        elif hasattr(b[k],'keys') and ('$in' in b[k] or '$nin' in b[k] or '$ne' in b[k]):
            pass
        else:
            a[k] = b[k]

def generate_splits(dbname,collectionname,task_query,N,ntrain,ntest,
                    ntrain_pos = None, ntest_pos = None, universe = None,
                    use_negate = False, overlap=None):

    task_query = copy.deepcopy(task_query)
    print('Generating splits ...')
    if universe is None:
        universe = SON([])

    connection = pm.Connection(document_class=SON)
    db = connection[dbname]
    data = db[collectionname + '.files']

    fs = gridfs.GridFS(db,collection=collectionname)

    combine_things(task_query,universe)
    
    print('T',task_query)
    task_data = list(data.find(task_query,fields=["filename"]))
    task_fnames = [str(x['filename']) for x in task_data]
    N_task = len(task_data)
    
    if use_negate:
        task_fnames = np.array(task_fnames)
        all_data = list(data.find(universe,fields=["filename"]))
        all_fnames = np.array([str(x['filename']) for x in all_data])
        I = np.invert(tb.isin(all_fnames,task_fnames)).nonzero()[0]
        nontask_data = [all_data[ind] for ind in I]
        nontask_fnames = [str(x['filename']) for x in nontask_data]
        assert set(task_fnames).intersection(nontask_fnames) == set([]), set(task_fnames).intersection(nontask_fnames)
    else:
        nontask_query = {'filename':{'$nin':task_fnames}}
        nontask_query.update(universe)
        nontask_data = get_most_recent_files(data,nontask_query)
        
    N_nontask = len(nontask_data)

    assert ntrain + ntest <= N_task + N_nontask, "Not enough training and/or testing examples " + str([N_task,N_nontask])
    
        
    splits = []  
    for ind in range(N):
        print('... split', ind)
        if ntrain_pos is not None:
            ntrain_neg = ntrain - ntrain_pos
            assert ntrain_pos <= N_task, "Not enough positive training examples, there are: " + str(N_task)
            assert ntrain_neg <= N_nontask, "Not enough negative training examples, there are: " + str(N_nontask)
            
            perm_pos = sp.random.permutation(len(task_data))
            perm_neg = sp.random.permutation(len(nontask_data))
            
            train_data = [task_data[i] for i in perm_pos[:ntrain_pos]] + [nontask_data[i] for i in perm_neg[:ntrain_neg]]    
            
            if ntest_pos is not None:
                ntest_neg = ntest - ntest_pos
                assert ntest_pos <= N_task - ntrain_pos, "Not enough positive test examples, there are: " + str(N_task - ntrain_pos)
                assert ntest_neg <= N_nontask - ntrain_neg, "Not enough negative test examples, there are: " + str(N_nontask - ntrain_neg)       
                test_data = [task_data[i] for i in perm_pos[ntrain_pos:ntrain_pos + ntest_pos]] + [nontask_data[i] for i in perm_neg[ntrain_neg:ntrain_neg + ntest_neg]]          
            else:     
                nontrain_data = [task_data[i] for i in perm_pos[ntrain_pos:]] + [nontask_data[i] for i in perm_neg[ntrain_neg:]]
                new_perm = sp.random.permutation(len(nontrain_data))
                test_data = [nontrain_data[i] for i in new_perm[:ntest]]
            
        
        else:
            if ntest_pos is not None:
                ntest_neg = ntest - ntest_pos
                assert ntest_pos <= N_task, "Not enough positive test examples, there are: " + str(N_task)
                assert ntest_neg <= N_nontask, "Not enough negative test examples, there are: " + str(N_nontask)                   
                perm_pos = sp.random.permutation(len(task_data))
                perm_neg = sp.random.permutation(len(nontask_data))
                test_data = [task_data[i] for i in perm_pos[:ntest_pos]] + [nontask_data[i] for i in perm_neg[:ntest_neg]]   
                nontest_data = [task_data[i] for i in perm_pos[ntest_pos:]] + [nontask_data[i] for i in perm_neg[ntest_neg:]]
                new_perm = sp.random.permutation(len(nontest_data))
                train_data = [nontest_data[i] for i in new_perm[:ntrain]]               
            else:
                all_data = task_data + nontask_data
                perm = sp.random.permutation(len(all_data))
                train_data = [all_data[i] for i in perm[:ntrain]]
                test_data = [all_data[i] for i in perm[ntrain:ntrain + ntest]]
            
        train_filenames = np.array([str(_t['filename']) for _t in train_data])
        test_filenames = np.array([str(_t['filename']) for _t in test_data])
        
        train_labels = tb.isin(train_filenames,task_fnames)
        test_labels = tb.isin(test_filenames,task_fnames)
         
        #train_labels = sp.array([x['filename'] in task_fnames for x in train_data])
        #test_labels = sp.array([x['filename'] in task_fnames for x in test_data])

        assert set(train_filenames).intersection(test_filenames) == set([]), str(set(train_filenames).intersection(test_filenames))
        
        split = {'train_data': train_data, 'test_data' : test_data, 'train_labels':train_labels,'test_labels':test_labels}
        splits.append(split)
   
    return splits


def validate(idseq):
    ids = ListUnion(idseq)
    ids1 = [id[1] for id in ids]
    assert len(uniqify(ids1)) == sum([len(X) for X in idseq]), 'Classes are not disjoint.'
    return ids
    
    
 
def generate_multi_splits(dbname, collectionname, task_queries, N, ntrain,
                          ntest, universe=None, labels=None, overlap=None,
                          balance = None, kfold=None):

    nq = len(task_queries)
    if labels is None:
        labels = range(nq)

    task_queries = [copy.deepcopy(task_query) for task_query in task_queries]
    print('Generating splits ...')
    if universe is None:
        universe = SON([])

    connection = pm.Connection(document_class=SON)
    db = connection[dbname]
    data = db[collectionname + '.files']

    fs = gridfs.GridFS(db,collection=collectionname)

    for task_query in task_queries:
        combine_things(task_query,universe)

    task_data = [list(data.find(task_query,fields=["filename"])) for task_query in task_queries]

    if kfold is not None:
        return generate_multi_kfold_splits(labels,kfold,task_data)
    else:
        return generate_multi_random_splits(labels,ntrain,ntest,task_data,overlap,N,balance)

def generate_multi_kfold_splits(labels,kfold,task_data):

    total = sum([len(td) for td in task_data])
    perms = [sp.random.permutation(len(td)) for td in task_data]

    indsets = []
    for td in task_data:
        indset = []
        l = len(td)
        ld = l/kfold
        for i in range(kfold-1):
            indset.append(range(ld*i,ld*(i+1)))
        indset.append(range(ld*(kfold-1),l))
        indsets.append(indset)

    splits = []
    for ind in range(kfold):
        l_ind_out = range(kfold)
        l_ind_out.pop(ind)
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        for indset, td, p, label in zip(indsets,task_data,perms,labels):
            for _ind in l_ind_out:
                new = [td[p[i]] for i in indset[_ind]]
                train_data.extend(new)
                train_labels.extend([label]*len(new))
            new = [td[p[i]] for i in indset[ind]]
            test_data.extend(new)
            test_labels.extend([label]*len(new))

        train_filenames = np.array([str(_t['filename']) for _t in train_data])
        test_filenames = np.array([str(_t['filename']) for _t in test_data])
        assert set(train_filenames).intersection(test_filenames) == set([]), str(set(train_filenames).intersection(test_filenames))
        assert len(train_data) + len(test_data) == total

        split = {'train_data': train_data, 'test_data' : test_data, 'train_labels':train_labels,'test_labels':test_labels}
        splits.append(split)

    test_datas = [split['test_data'] for split in splits]
    test_filenames = [[str(_t['filename']) for _t in tds] for tds in test_datas]
    all_test_filenames = list(itertools.chain(*test_filenames))
    assert len(all_test_filenames) == total
    assert len(np.unique(all_test_filenames)) == total

    return splits
        
def generate_multi_random_splits(labels,ntrain,ntest,task_data,overlap,N,balance):
    
    floor = math.floor
    
    if balance is None:
        ntrain_vec = [ntrain/nq]*(nq - 1) + [ntrain - (ntrain/nq)*(nq-1)]
        ntest_vec = [ntest/nq]*(nq - 1) + [ntest - (ntest/nq)*(nq-1)]
    else:
        assert len(balance) == nq - 1
        ntrain_vec = [int(floor(ntrain*b)) for b in balance]
        ntrain_vec = ntrain_vec + [ntrain - sum(ntrain_vec)]
        ntest_vec = [int(floor(ntest*b)) for b in balance]
        ntest_vec = ntest_vec + [ntest - sum(ntest_vec)]
    
    for (tq,td,ntr,nte) in zip(task_queries,task_data,ntrain_vec,ntest_vec):
        assert ntr + nte <= len(td), 'not enough examples to train/test for %s, %d needed, but only have %d' % (repr(tq),ntr+nte,len(td))

    
    if overlap:
        assert isinstance(overlap,float) and 0 < overlap <= 1, 'overlap must be a float in (0,1]'
        for (_i,(td,ntr,nte)) in enumerate(zip(task_data,ntrain_vec,ntest_vec)):
            eff_n = int(math.ceil((ntr+nte)*(1/overlap)))
            #add given category intersections and check for error
            perm = sp.random.permutation(len(td))
            task_data[_i] = [td[_j] for _j in perm[:eff_n]]
    
    splits = []
    for ind in range(N):
        print('... split', ind)
    
        train_data = []
        test_data = []
        train_labels = []
        test_labels = []
        for (label,td,ntr,nte) in zip(labels,task_data,ntrain_vec,ntest_vec):
            td = get_correct_td(td,test_data)
            assert len(td) >= ntr + nte, 'problem with %s, need %d have %d'  % (label,ntr+nte,len(td))
            perm = sp.random.permutation(len(td))
            train_data.extend([td[i] for i in perm[:ntr]])
            td = get_correct_td(td,train_data)
            assert len(td) >= nte, 'problem with %s test, need %d have %d'  % (label,nte,len(td))
            perm = sp.random.permutation(len(td))
            test_data.extend([td[i] for i in perm[:nte]])
            train_labels.extend([label]*ntr)
            test_labels.extend([label]*nte)
    
        train_filenames = np.array([str(_t['filename']) for _t in train_data])
        test_filenames = np.array([str(_t['filename']) for _t in test_data])
        assert set(train_filenames).intersection(test_filenames) == set([]), str(set(train_filenames).intersection(test_filenames))
             
        split = {'train_data': train_data, 'test_data' : test_data, 'train_labels':train_labels,'test_labels':test_labels}
        splits.append(split)
   
    return splits


def get_correct_td(td,T):
    tf = np.array([str(t['filename']) for t in td])
    Tf = np.array([str(t['filename']) for t in T])
    good_inds = np.invert(tb.fast.isin(tf,Tf)).nonzero()[0]
    return [td[ind] for ind in good_inds]
