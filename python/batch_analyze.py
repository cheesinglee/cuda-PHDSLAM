# -*- coding: utf-8 -*-

from numpy import *
from matplotlib import pyplot
from ospa import ospa_distance
from scipy.io import loadmat
from RangeBearingMeasurementModel import *
import cProfile
import pstats
import os
import cPickle
import fnmatch
from plot_results import plot_errors


def compute_error_k(logfilename,true_pose,true_map):
    f = open(logfilename,'r')
    est_pose = fromstring(f.readline(),sep=' ')[0:3]
    est_map = fromstring(f.readline(),sep=' ')
    log_particle_weights = fromstring( f.readline(), sep=' ' )
    map_weights = est_map[0::7]
    map_x = est_map[1::7]
    map_y = est_map[2::7]
    map_means = hstack((map_x[:,newaxis],map_y[:,newaxis]))
    if map_means.size > 0:
        w_sum = round(sum(map_weights))
        sort_idx = argsort(map_weights)[::-1]
        map_means = map_means[sort_idx[:w_sum],:]
    pose_err = sqrt(sum( (true_pose[:2] - est_pose[:2])**2))
    ospa_tuple = ospa_distance( true_map, map_means, p=1, c=5 )
    ospa_err = ospa_tuple[0]
    ospa_loc = ospa_tuple[1]
    ospa_cn = ospa_tuple[2]
    
    n_eff = 1/sum(exp(log_particle_weights)**2)
    f.close()
    return (pose_err,ospa_err,ospa_loc,ospa_cn,n_eff)
    

def compute_pose_error(true_pose,est_pose):
    return sqrt(sum( (true_pose[:2] - est_pose[:2])**2))


def compute_error(basedir,ground_truth):
    n_steps = ground_truth['n_steps']
    true_maps = ground_truth['true_maps']
    true_traj = ground_truth['true_traj']
    data_dirs = []
    for (dirpath,dirnames,filenames) in os.walk(basedir):
        n_files = len(filenames)
        pattern = ['state_estimate*.log']*n_files
        if any(map(fnmatch.fnmatch,filenames,pattern)):
            data_dirs.append(dirpath)
    n_runs = len(data_dirs)
    pose_err = zeros([n_steps,n_runs])
    map_err = zeros([n_steps,n_runs])
    loc_err = zeros([n_steps,n_runs])
    cn_err = zeros([n_steps,n_runs])
    n_eff = zeros([n_steps,n_runs])
    print 'n_steps = %d\tn_runs = %d' % (n_steps,n_runs)
    for k in xrange(n_steps):
        file_list = []
        filename_k = 'state_estimate{:05d}.log'.format(k)
        file_list = map(os.path.join,data_dirs,[filename_k]*n_runs)
        print(k)
#        poses_k = empty((n_runs,3))
#        maps_k = []
        true_pose_list = [true_traj[:,k]]*n_runs
        true_map_list = [true_maps[k].transpose()]*n_runs
        results = map(compute_error_k,file_list,true_pose_list,true_map_list)
        (pose_err_tmp,map_err_tmp,loc_err_tmp,cn_err_tmp,n_eff_tmp) = zip(*results)
        pose_err[k,:] = asarray(pose_err_tmp)
        map_err[k,:] = asarray(map_err_tmp)
        loc_err[k,:] = asarray(loc_err_tmp)
        cn_err[k,:] = asarray(cn_err_tmp)
        n_eff[k,:] = asarray(n_eff_tmp)
#        p_list = [1]*n_runs
#        c_list = [20]*n_runs
#        for i in xrange(n_runs):
#            f = open(file_list[i],'r')
#            pose_estimate = fromstring( f.readline(), sep=' ' )[0:3]
#            map_estimate = fromstring( f.readline(), sep=' ' )
#            map_weights = map_estimate[0::7]
#            map_x = map_estimate[1::7]
#            map_y = map_estimate[2::7]
#            map_means = hstack((map_x[:,newaxis],map_y[:,newaxis]))
#            if map_means.size > 0:
#                w_sum = round(sum(map_weights))
#                sort_idx = argsort(map_weights)[::-1]
#                map_means = map_means[sort_idx[:w_sum],:]
#            poses_k[i,:] = pose_estimate
#            maps_k.append(map_means)           
##            pose_err[k] += sqrt(sum( (true_traj[:2,i] - pose_estimate[:2])**2))/n_runs
#            map_err[k] += ospa_distance( true_maps[k].transpose(), map_means, p=1, c=20 )/n_runs
#        pose_err[k,:] = asarray(map( compute_pose_error, true_pose_list, poses_k ) )
#        map_err[k,:] = asarray(map( ospa_distance, true_map_list, maps_k, p_list, c_list ) )
    return (pose_err,map_err,loc_err,cn_err,n_eff)


if __name__ == "__main__":
    import tkFileDialog
    matfilename = tkFileDialog.askopenfilename(filetypes=[('mat files','.mat')])
    if len(matfilename) == 0:
        print 'user cancelled selection'
        exit()
    basedir = os.path.dirname(matfilename)
    
    print('loading mat-file...')
    sim = loadmat(matfilename)['sim']
    trajectory = sim['traj'][0,0]
    ground_truth = sim['groundTruth'][0,0]
    n_steps = sim['n_steps'][0,0][0,0]

    true_maps = []    
    for k in xrange(n_steps):
        loc = ground_truth[0,k][0,0][0]
        true_maps.append(loc)

    ground_truth = {'n_steps':n_steps,'true_maps':true_maps,'true_traj':trajectory}

#    basedir = '/home/cheesinglee/workspace/cuda-PHDSLAM/batch_results/ackerman3_1cluster/'
    cProfile.run('results=compute_error(basedir,ground_truth)','compute_error_prof')
    stats = pstats.Stats('compute_error_prof')
    stats.sort_stats('cumulative').print_stats()
    results_file = open('results.pk2','wb')
    cPickle.dump(results,results_file,cPickle.HIGHEST_PROTOCOL)
    results_file.close()
    plot_errors(['results.pk2'])
        
        



