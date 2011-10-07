# -*- coding: utf-8 -*-

import cPickle
import tables
from numpy import *
from RangeBearingMeasurementModel import *
import os
from matplotlib import pyplot

motion_params = {
    'std_encoder':2,
    'std_alpha':0.0873,
    'l':2.83,
    'h':0.76,
    'a':3.78,
    'b':0.50}
    
sensor_params = {
    'max_range':10,
    'max_bearing':pi,
    'std_range':1.0,
    'std_bearing':0.0349,
    'pd':0.95,
    'clutter_rate':20}
measurement_model = RangeBearingMeasurementModel(sensor_params)

print('loading mat-file...')
file = tables.openFile('groundtruth.mat')
landmarks = file.root.staticMap[:].transpose()
trajectory = file.root.traj[:].transpose()
controls = file.root.controls[:].transpose()
file.close()

n_runs = 50 

print('generating simulation inputs...')
for n in xrange(n_runs):
    print(n)
    # make the noisy control inputs
    n_controls = size(controls,1)
    encoder_noise = random.randn(n_controls)*motion_params['std_encoder']
    steering_noise = random.randn(n_controls)*motion_params['std_alpha']
    controls_noisy = controls + vstack((encoder_noise,steering_noise))
    controls_noisy[1,:] = wrap_angle(controls_noisy[1,:])
    
    # generate noisy measurements
    n_steps = size(trajectory,1)
    measurements = [] ;
    for i in xrange(n_steps):
        z = measurement_model.compute_noisy_measurement(trajectory[:,i],landmarks)
        measurements.append(z)
    
    # write to text file
    os.mkdir(str(n))    
    dir_str = str(n)+os.sep 
    controls_file = open(dir_str+'controls_synth.txt','w')
    controls_file.write('% velocity\tsteering angle\n')
    for i in xrange(n_controls):
        control_str = '{u[0]} {u[1]}\n'.format(u=controls_noisy[:,i])
        controls_file.write(control_str)
    controls_file.close()
    
    measurements_file = open(dir_str+'measurements_synth.txt','w')
    measurements_file.write('% measurements from simulation data. One time step per line, each pair of of numbers is a range/bearing measurement.\n')
    for i in xrange(n_steps):
        Z = measurements[i] ;
        n_z = size(Z,1)
        z_str = ''
        for j in xrange(n_z):
            z_str += '{z[0]} {z[1]} '.format(z=Z[:,j])
        z_str += '\n'
        measurements_file.write(z_str)
    measurements_file.close()
    
    datafile = open(dir_str+'simdata.pk2','w')
    cPickle.dump( landmarks, datafile, cPickle.HIGHEST_PROTOCOL )
    cPickle.dump( trajectory, datafile, cPickle.HIGHEST_PROTOCOL )
    cPickle.dump( controls, datafile, cPickle.HIGHEST_PROTOCOL )
    cPickle.dump( controls_noisy, datafile, cPickle.HIGHEST_PROTOCOL )
    cPickle.dump( measurements, datafile, cPickle.HIGHEST_PROTOCOL )
    datafile.close()
    





