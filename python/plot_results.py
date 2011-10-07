# -*- coding: utf-8 -*-

import cPickle
from matplotlib import pyplot
from numpy import *

def plot_errors(filenames):
    colors = 'brgymc'
    styles = ['-','--','-.',':']
    idx = 0
    for filename in filenames:
        file = open(filename,'rb')
        results = cPickle.load(file)
        file.close()
        pose_err = results[0]
        map_err = results[1]
        loc_err = results[2] 
        cn_err = results[3]
        n_eff = results[4]
        n_steps = pose_err.shape[0]
        mean_pose_err = mean(pose_err,axis=1)
        std_pose_err = std(pose_err,axis=1)
        mean_map_err = mean(map_err,axis=1)
        std_map_err = std(map_err,axis=1)
        mean_loc_err = mean(loc_err,axis=1)
        std_loc_err = std(loc_err,axis=1)
        mean_cn_err = mean(cn_err,axis=1)
        std_cn_err = std(cn_err,axis=1)
        
        mean_n_eff = mean(n_eff,axis=1)
        std_n_eff = std(n_eff,axis=1)
        interval = n_steps/5
        pyplot.figure(1)
        pyplot.plot(arange(n_steps),mean_pose_err,linewidth=2,color=colors[idx],
                    linestyle=styles[idx])
        pyplot.errorbar(arange(0,n_steps,interval),mean_pose_err[0::interval],
                        yerr=std_pose_err[0::interval],ecolor=colors[idx],
                        fmt=None,elinewidth=2,capsize=6)
        pyplot.ylabel('Pose Error (meters)',fontsize='large')
        pyplot.xlabel('Time Step',fontsize='large')
        pyplot.tick_params(labelsize='medium')
        
        pyplot.figure(2)
        pyplot.subplot(311)
        pyplot.plot(arange(n_steps),mean_map_err,color=colors[idx],linewidth=2,
                    linestyle=styles[idx])
        pyplot.errorbar(arange(0,n_steps,interval),mean_map_err[0::interval],
                        yerr=std_map_err[0::interval],ecolor=colors[idx],
                        fmt=None,elinewidth=2,capsize=6)
        pyplot.ylabel('OSPA Error',fontsize='large')
        pyplot.subplot(312)
        pyplot.plot(arange(n_steps),mean_loc_err,color=colors[idx],linewidth=2,
                    linestyle=styles[idx])
        pyplot.errorbar(arange(0,n_steps,interval),mean_loc_err[0::interval],
                        yerr=std_loc_err[0::interval],ecolor=colors[idx],
                        fmt=None,elinewidth=2,capsize=6)
        pyplot.ylabel('Localization Error',fontsize='large')
        pyplot.subplot(313)
        pyplot.plot(arange(n_steps),mean_cn_err,color=colors[idx],linewidth=2,
                    linestyle=styles[idx])
        pyplot.errorbar(arange(0,n_steps,interval),mean_cn_err[0::interval],
                        yerr=std_cn_err[0::interval],ecolor=colors[idx],
                        fmt=None,elinewidth=2,capsize=6)
        pyplot.ylabel('Cardinality Error',fontsize='large')
        pyplot.xlabel('Time Step',fontsize='large')
        pyplot.tick_params(labelsize='medium')
        
#        pyplot.figure(3)
#        pyplot.plot(arange(n_steps),mean_n_eff,linewidth=2,color=colors[idx],
#                    linestyle=styles[idx])
#        pyplot.errorbar(arange(0,n_steps,interval),mean_n_eff[0::interval],
#                        yerr=std_n_eff[0::interval],ecolor=colors[idx],
#                        fmt=None,elinewidth=2,capsize=6)
##        pyplot.axis('scaled')
#        pyplot.ylabel('Effective Sample Size',fontsize='large')
#        pyplot.xlabel('Time Step',fontsize='large')
#        pyplot.tick_params(labelsize='medium')
        
        idx += 1
    pyplot.show()
    
def plot_map(basedir,ground_truth):
    import glob
    import os
    n_steps = ground_truth['n_steps']
    est_traj = empty([3,n_steps])
    # compute the estimated trajectory
    for k in xrange(n_steps):
        filename = os.path.join(basedir,'state_estimate'+str(k)+'.log')
        f = open(filename,'r')
        est_traj[:,k] = fromstring( f.readline(), sep=' ' )[0:3]
        k += 1
        f.close()
        
    f = open(filename,'r')
    f.readline()
    est_map = fromstring( f.readline(), sep=' ' )
    map_weights = est_map[0::7]
    map_x = est_map[1::7]
    map_y = est_map[2::7]
    map_means = vstack((map_x,map_y))
    if map_means.size > 0:
        w_sum = round(sum(map_weights))
        sort_idx = argsort(map_weights)[::-1]
        map_means = map_means[:,sort_idx[:w_sum]]
    pyplot.plot(ground_truth['trajectory'][0,0:n_steps],
                ground_truth['trajectory'][1,0:n_steps],'k')
    pyplot.plot(est_traj[0,0:n_steps],est_traj[1,0:n_steps],'r--')
    pyplot.plot(ground_truth['true_maps'][n_steps-1][0,:],
                ground_truth['true_maps'][n_steps-1][1,:],'b*')
    pyplot.plot(map_means[0,:],map_means[1,:],'bo')
    pyplot.axis('scaled')
    pyplot.show()
    
if __name__ == "__main__":
    import tkFileDialog
    
    filenames = tkFileDialog.askopenfilenames()
    plot_errors(filenames)
    
#    import tables
#    from RangeBearingMeasurementModel import *
#    print('loading mat-file...')
#    file = tables.openFile('groundtruth.mat')
#    landmarks = file.root.staticMap[:].transpose()
#    trajectory = file.root.traj[:].transpose()
#    file.close()
##    n_steps = trajectory.shape[1]
#    n_steps = 400 
#    n_landmarks = landmarks.shape[1]
#    observed = zeros(n_landmarks,dtype=bool)
#    true_maps = []
#    sensor_params = {
#        'max_range':10,
#        'max_bearing':pi,
#        'std_range':1.0,
#        'std_bearing':0.0349,
#        'pd':0.95,
#        'clutter_rate':20}
#    measurement_model = RangeBearingMeasurementModel(sensor_params)
#    for k in xrange(n_steps):
#        # check if we have already seen everything
#        if all( observed ):
#            features = landmarks
#        else:
#            pose = trajectory[:,k]
#            in_range = measurement_model.check_in_range(pose,landmarks)
#            observed = logical_or(observed,in_range)
#            features = landmarks[:,observed]
#        true_maps.append(features)
#    ground_truth = {'n_steps':n_steps,'true_maps':true_maps,
#                    'trajectory':trajectory}
#                
#    basedir = tkFileDialog.askdirectory()
#    plot_map(basedir,ground_truth)
    

    