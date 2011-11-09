import subprocess
import re
import tempfile
import cPickle
import string
import os
import time
import hashlib

import BeautifulSoup

SGE_STATUS_INTERVAL = 5
SGE_EXIT_STATUS_PATTERN = re.compile('exit_status[\s]*([\d])')


class QacctParsingError(Exception):
    def __init__(self,job,name):
        self.msg = "Can't parse exit status from " + repr(name) + " for job " + repr(job) + "."


def wait_and_get_statuses(joblist):

    f = tempfile.NamedTemporaryFile(delete=False)
    name = f.name
    f.close()
    
    jobset =  set(joblist)
    
    statuses = []
    while True:
        os.system('qstat -xml > ' + name)
        Soup = BeautifulSoup.BeautifulStoneSoup(open(name))
        running_jobs = [str(x.contents[0]) for x in Soup.findAll('jb_job_number')]
        if jobset.intersection(running_jobs):
            time.sleep(SGE_STATUS_INTERVAL)
        else:
            break
    
    
    for job in jobset:
        e = os.system('qacct -j ' + job + ' > ' + name)
        if e != 0:
            time.sleep(20)
        os.system('qacct -j ' + job + ' > ' + name)
        s = open(name).read()      
        try:
            res = SGE_EXIT_STATUS_PATTERN.search(s)
            child_exitStatus = int(res.groups()[0])
            statuses.append(child_exitStatus)
        except:
            raise QacctParsingError(job,name)
        else:
            pass
    
    os.remove(name)
    return statuses


def callfunc(fn,argfile):
    args = cPickle.loads(open(argfile).read())
    
    if isinstance(args,list):
        pos_args = args[0]
        kwargs = args[1]
    elif isinstance(args,dict):
        pos_args = ()
        kwargs = args
    else:
        pos_args = args
        kwargs = {}

    os.remove(argfile)    
    fn(*pos_args,**kwargs)
    
SGE_SUBMIT_PATTERN = re.compile("Your job ([\d]+) ")

import random

def get_temp_file():
    random.seed()
    hash = "%016x" % random.getrandbits(128)
    filename = os.path.join(os.environ['HOME'] , 'qsub_tmp',"qsub_" + hash)
    return open(filename,'w')

def qsub(fn, args, opstring='', python_executable='python'):

    module_name = fn.__module__
    fnname = fn.__name__
    
    f = get_temp_file()
    argfile = f.name
    cPickle.dump(args,f)
    f.close()

    f = get_temp_file()
    scriptfile = f.name
    call_script = string.Template(call_script_template).substitute({'MODNAME':module_name,
                                                         'FNNAME':fnname,
                                                         'ARGFILE':argfile,
                                                         'PYEXEC':python_executable})
    f.write(call_script)
    f.close()
    
    p = subprocess.Popen('qsub ' + opstring + ' ' + scriptfile,shell=True,stdout=subprocess.PIPE)
    sts = os.waitpid(p.pid,0)[1]

    if sts == 0:
        output = p.stdout.read()
        jobid = SGE_SUBMIT_PATTERN.search(output).groups()[0]
    else:
        raise 

    os.remove(scriptfile)
    
    return jobid
    
call_script_template = """#!/bin/bash
#$$ -V
#$$ -cwd
#$$ -S /bin/bash

$PYEXEC -c "import $MODNAME, sge_utils; sge_utils.callfunc($MODNAME.$FNNAME,'$ARGFILE')"

"""
