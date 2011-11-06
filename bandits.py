"""
"""
import numpy
from hyperopt.genson_bandits import GensonBandit

import evaluation
from dbutils import son_escape

def fill_in_default(a,b):
    if hasattr(a,'keys'):
        assert hasattr(b,'keys')
        for k in b.keys():
            if k not in a:
                a[k] = b[k]
            else:
                fill_in_default(a[k],b[k])
    elif hasattr(a,'iter'):
        assert len(a) == len(b)
        for (aa,bb) in zip(a,b):
            fill_in_default(aa,bb)


class EvaluationBandit(GensonBandit):

    def __init__(self, source_string,
                 image_hash,
                 model_hash,
                 image_config_gen,
                 opt_hash,
                 convolve_func_name,
                 use_db = False):
    
        super(EvaluationBandit,self).__init__(source_string=source_string)
        
        connection = pm.Connection(document_class = SON)
        db = connection[DB_NAME]
        self.perf_col = db['performance']

        self.image_hash = image_hash
        self.image_config_gen = image_config_gen
        self.opt_hash = opt_hash
        self.convolve_func_name = convolve_func_name
        self.use_db = use_db
        self.model_hash = model_hash
    
    def evaluate(self,argd,ctrl):
        model,task = self.model_and_task_from_template(argd)
        performance = self.get_performance(model,task)
        if performance:
            status = 'ok'
        else:
            status = '?'  #what here?
        return dict(loss = performance, status = status)

    def model_task_from_template(self,template):
        model = template['model']
        default = template['default_model']
        fill_in_default(model, default)
        task = template['task']
        return model, task
        
    def get_performance(self,model,task):
    
        opt_hash = self.opt_hash
        convolve_func_name = self.convolve_func_name
        image_hash = self.image_hash
        image_config_gen = self.image_config_gen
        use_db = self.use_db
        
        opstring= '-l qname=hyperopt.q -o /home/render/hyperopt_jobs -e /home/render/hyperopt_jobs'
        jobid = qsub(evaluation.get_performance,
                     (model,task,image_hash,image_config_gen,opt_hash),
                     opstring=opstring)
        
        status = wait_and_get_statuses([job_id])

        if status == 0:
            perf = self.perf_col.find_one({'image_hash':image_hash,
                               'model':model,
                               'task':son_escape(task),
                               '__hash__':opt_hash})                                   
            return perf['test_accuracy']
    


                    
     
