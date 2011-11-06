import scipy as sp
import numpy as np
from scikits.learn import svm as sklearn_svm
from scikits.learn import linear_model as sklearn_linear_model
from scikits.learn.linear_model.logistic import LogisticRegression

'''
SVM classifier module
'''

def multi_classify(train_features,
                     train_labels,
                     test_features,
                     test_labels,
                     classifier_kwargs,
                     relabel = True):
    """
    Classifier using the built-in multi-class classification capabilities of liblinear
    """

    if relabel:
        labels = sp.unique(sp.concatenate((train_labels, test_labels)))
        label_to_id = dict([(k,v) for v, k in enumerate(labels)]) 
        train_ids = sp.array([label_to_id[i] for i in train_labels])
    else:
        train_ids = train_labels
        labels = None
    
    classifier,train_mean, train_std, was_sphered = classifier_train(train_features, train_ids, test_features, **classifier_kwargs)
    
    weights = classifier.coef_.T
    bias = classifier.intercept_

        
    test_prediction = labels[classifier.predict(test_features)]
    test_accuracy = float(100*(test_prediction == test_labels).sum() / float(len(test_prediction)))
    train_prediction = labels[classifier.predict(train_features)]
    train_accuracy = float(100*(train_prediction == train_labels).sum() / float( len(train_prediction)))
    
    margin_fn = lambda v : (sp.dot(v,weights) + bias)
    test_margins = margin_fn(test_features)
#    test_margin_prediction = labels[test_margins.argmax(1)]
#    train_margins = margin_fn(train_features)
#    train_margin_prediction = labels[train_margins.argmax(1)]
#    assert (test_prediction == test_margin_prediction).all(), 'test margin prediction not correct'
#    assert (train_prediction == train_margin_prediction).all(), 'train margin prediction not correct'    

    test_prediction = classifier.predict(test_features)
    train_prediction = classifier.predict(train_features)
    if labels:
        test_prediction = labels[test_prediction]
        train_prediction = labels[train_prediction]
    
    cls_data = {'coef' : weights, 
     'intercept' : bias, 
     'train_labels': train_labels,
     'test_labels' : test_labels,
     'train_prediction': train_prediction, 
     'test_prediction' : test_prediction,
     'labels' : labels,
     'train_mean' : train_mean,
     'train_std' : train_std,
     'sphere' : was_sphered,
     'test_margins' : test_margins,
#     'train_margins' : train_margins
     }


    return {'cls_data' : cls_data,
     'train_accuracy' : train_accuracy,
     'test_accuracy' : test_accuracy,
     'mean_ap' : mean_ap,
     'mean_auc' : mean_auc
     }
     
    stats = multiclass_stats(test_labels,test_prediction,train_labels,train_prediction,labels)     

    result = {'cls_data':cls_data}
    result.update(stats)
    return result
    

def multiclass_stats(test_actual,test_predicted,train_actual,train_predicted,labels):
    test_accuracy = float(100*(test_prediction == test_labels).sum() / float(len(test_prediction)))
    train_accuracy = float(100*(train_prediction == train_labels).sum() / float( len(train_prediction)))
    train_aps = []
    test_aps = []
    train_aucs = []
    test_aucs = []
    if len(labels) == 2:
        labels = labels[1:]
    for label in labels:
        train_prec,train_rec = precision_and_recall(train_actual,train_predicted,label)
        test_prec,test_rec = precision_and_recall(test_actual,test_predicted,label)
        train_ap = ap_from_prec_and_rec(train_prec,train_rec)
        test_ap = ap_from_prec_and_rec(test_prec,test_rec)
        train_aps.append(train_ap)
        test_aps.append(test_ap)
        train_auc = auc_from_prec_and_rec(train_prec,train_rec)
        test_auc = auc_from_prec_and_rec(test_prec,test_rec)
        train_aucs.append(train_auc)
        test_aucs.append(test_auc)
    train_ap = np.array(train_aps).mean()
    test_ap = np.array(test_aps).mean()
    train_auc = np.array(train_aucs).mean()
    test_auc = np.array(test_aucs).mean()
    return {'train_accuracy' : train_accuracy,
            'test_accuracy' : test_accuracy,
            'train_ap' : train_ap,
            'test_ap' : test_ap,
            'train_auc' : train_auc,
            'test_auc' : test_auc}

def precision_and_recall(actual,predicted,cls):
    c = (actual == cls)
    si = sp.argsort(-c)
    tp = sp.cumsum(sp.single(predicted[si] == cls))
    fp = sp.cumsum(sp.single(predicted[si] != cls))
    rec = tp /sp.sum(predicted == cls)
    prec = tp / (fp + tp)
    return prec,rec

    
def ap_from_prec_and_rec(prec,rec):
    ap = 0
    rng = sp.arange(0, 1.1, .1)
    for th in rng:
        parray = prec[rec>=th]
        if len(parray) == 0:
            p = 0
        else:
            p = parray.max()
        ap += p / rng.size
    return ap


def auc_from_prec_and_rec(prec,rec):
    #area under curve
    h = sp.diff(rec)
    auc = sp.sum(h * (prec[1:] + prec[:-1])) / 2.0
    return auc        
    
    
def classifier_train(train_features,
                     train_labels,
                     test_features,
                     classifier_type = "liblinear",
                     sphere = True,
                     **kwargs
                     ):
    """ Classifier training using SVMs

    Input:
    train_features = training features (both positive and negative)
    train_labels = corresponding label vector
    svm_eps = eps of svm
    svm_C = C parameter of svm
    classifier_type = liblinear or libsvm"""
       
    #sphering
    if sphere:
        train_features, test_features,fmean,fstd = __sphere(train_features, test_features)
    else:
        fmean = None
        fstd = None

    if classifier_type == 'liblinear':
        clf = sklearn_svm.LinearSVC(**kwargs)
    if classifier_type == 'libSVM':
        clf = sklearn_svm.SVC(**kwargs)
    elif classifier_type == 'LRL':
        clf = LogisticRegression(**kwargs)
    elif classifier_type == 'MCC':
        clf = CorrelationClassifier(**kwargs)
    elif classifier_type.startswith('svm.'):
        ct = classifier_type.split('.')[-1]
        clf = getattr(sklearn_svm,ct)(**kwargs)
    elif classifier_type.startswith('linear_model.'):
        ct = classifier_type.split('.')[-1]
        clf = getattr(sklearn_linear_model,ct)(**kwargs)
    
    clf.fit(train_features, train_labels)
    
    return clf,fmean, fstd, sphere

#sphere data
def __sphere(train_data, test_data):
    '''make data zero mean and unit variance'''

    fmean = train_data.mean(0)
    fstd = train_data.std(0)

    train_data -= fmean
    test_data -= fmean
    fstd[fstd==0] = 1
    train_data /= fstd
    test_data /= fstd

    return train_data, test_data, fmean, fstd
     
def max_predictor(weights,bias,labels):
    return lambda v : labels[(sp.dot(v,weights) + bias).argmax(1)]

def liblinear_predictor(clas, bias, labels):
    return lambda x : labels[liblinear_prediction_prediction_function(x,clas,labels)]

def liblinear_prediction_function(farray , clas, labels):

    if len(labels) > 2:
        nf = farray.shape[0]
        nlabels = len(labels)
        
        weights = clas.raw_coef_.ravel()
        nw = len(weights)
        nv = nw / nlabels
        
        D = np.column_stack([farray,np.array([.5]).repeat(nf)]).ravel().repeat(nlabels)
        W = np.tile(weights,nf)
        H = W * D
        H1 = H.reshape((len(H)/nw,nv,nlabels))
        H2 = H1.sum(1)
        predict = H2.argmax(1)
        
        return predict
    else:
    
        weights = clas.coef_.T
        bias = clas.intercept_
        
        return (1 - np.sign(np.dot(farray,weights) + bias) )/2
        

#=-=-=-=-=-=-=-=
#maximum correlation
#=-=-=-=-=-=-=-=

def uniqify(seq, idfun=None): 
    '''
    Relatively fast pure python uniqification function that preservs ordering
    ARGUMENTS:
        seq = sequence object to uniqify
        idfun = optional collapse function to identify items as the same
    RETURNS:
        python list with first occurence of each item in seq, in order
    '''
    try:

        # order preserving
        if idfun is None:
            def idfun(x): return x
        seen = {}
        result = []
        for item in seq:
            marker = idfun(item)
            # in old Python versions:
            # if seen.has_key(marker)
            # but in new ones:
            if marker in seen: continue
            seen[marker] = 1
            result.append(item)
    except TypeError:
        return [x for (i,x) in enumerate(seq) if x not in seq[:i]]
    else:
        return result


class CorrelationClassifier():

    def __init__(self):
        pass
        
    def fit(self,train_features,train_labels):
        self.labels = uniqify(train_labels)
        self.coef_ = np.array([train_features[train_labels == label].mean(0) for label in self.labels]).T
        self.intercept_ = -.5*(self.coef_ ** 2).sum(0)
        self.nums = [len((train_labels == label).nonzero()[0]) for label in self.labels]
             
    def predict(self,test_features):
        prediction = self.prediction_function(test_features)
        return [self.labels[i] for i in prediction]
          
    def prediction_function(self,test_features):
        return self.decision_function(test_features).argmax(1)
        
    def decision_function(self,test_features):
        return np.dot(test_features,self.coef_) + self.intercept_
        
    def update_fit(self,new_features,new_labels):
        unique_new_labels = uniqify(new_labels)
        for new_label in unique_new_labels: 
            new_f = new_features[new_labels == new_label]
            new_num = new_f.shape[0]
            if new_label in self.labels:
                l_ind = self.labels.index(new_label)
                num = self.nums[l_ind]
                self.coef_[:,l_ind] = (num * self.coef_[:,l_ind] + new_num * new_f.mean()) / (num + new_num)
                self.intercept_[l_ind] = -.5 * (self.coef_[:,l_ind] ** 2).sum()
                self.nums[l_ind] += new_num
            else:
                new_coef = np.empty((self.coef_.shape[0],self.coef_.shape[1] + 1))
                new_intercept = np.empty((self.intercept_.shape[0] + 1,))
                new_coef[:,:-1] = self.coef_
                new_intercept[:-1] = self.intercept_
                
                new_coef[:,-1] = new_f.mean()
                new_intercept[-1] = -.5 * (new_coef[:,-1] **2).sum()
                
                self.coef_ = new_coef
                self.intercept_ = new_intercept
                self.labels.append(new_label)
                self.nums.append(new_num) 
