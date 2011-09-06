# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 18:51:27 2011

@author: -
"""

import os
import re
import subprocess

n_runs = 50
project_dir = '/home/cheesinglee/workspace/cuda-PHDSLAM'
measurements_regex = r'(measurements_filename\s*=\s*).+'
controls_regex = r'(controls_filename\s*=\s*).+'

for n in xrange(n_runs):
    print 'run #: ',n    
    
    # set the working directory 
    os.chdir(project_dir+'/python/'+str(n))
    
    # modify the config file
    measurements_path = project_dir+'/python/'+str(n)+'/measurements_synth.txt'
    controls_path = project_dir+'/python/'+str(n)+'/controls_synth.txt'
    config_file = open(project_dir+'/cfg/config.cfg','rw+')
    config_str = config_file.read() 
    config_str = re.sub(measurements_regex,r'\1'+measurements_path,config_str)
    config_str = re.sub(controls_regex,r'\1'+controls_path,config_str)
    config_file.seek(0)
    config_file.write(config_str)
    config_file.close()

    # go!
    args = [project_dir+'/bin/cuda-PHDSLAM', project_dir+'/cfg/config.cfg']
    proc = subprocess.Popen(args)   
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError('Process returned with error!')
    
    
    