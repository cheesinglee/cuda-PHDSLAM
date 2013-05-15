# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 18:51:27 2011

@author: -
"""

import os
import shutil
import re
import subprocess

n_runs = 100
project_dir = '/home/cheesinglee/workspace/cuda-PHDSLAM'
data_dir = project_dir + os.path.sep + 'data/synth_mixed5'
data_regex = r'(data_directory\s*=\s*).+'

for n in xrange(1,n_runs+1):
    print 'run #: ',n    
    
    # set the working directory 
    os.chdir(data_dir+os.path.sep+str(n))
    
    # remove previous results
    contents = os.listdir(os.getcwd())
    for f in contents:
        if os.path.isdir(f):
            shutil.rmtree(f)
    
    # modify the config file
    sim_path = data_dir+os.path.sep+str(n)+os.path.sep
    config_file = open(project_dir+'/cfg/config.cfg','r')
    config_str = config_file.read() 
    config_file.close() ;
    config_str = re.sub(data_regex,r'\1'+sim_path,config_str)
    config_file = open(project_dir+'/cfg/config.cfg','w')
    config_file.write(config_str)
    config_file.close()
    
    # prepare the log file
    logfile = open('consoleout.log','w+')

    # go!
    args = [project_dir+'/bin/cuda-PHDSLAM', project_dir+'/cfg/config.cfg']
    proc = subprocess.Popen(args,stdout=logfile)   
    ret = proc.wait()
    logfile.close()
    if ret != 0:
        raise RuntimeError('Process returned with error!')
    
    
    
