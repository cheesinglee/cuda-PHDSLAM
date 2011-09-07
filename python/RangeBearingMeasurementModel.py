# -*- coding: utf-8 -*-

from numpy import *

def wrap_angle(a):
    rem = remainder(a,2*pi)
    rem[rem > pi] -= 2*pi
    rem[rem < -pi] += 2*pi
    return rem

class RangeBearingMeasurementModel:
    def __init__(self,params):
        # check that all the sensor parameters are present
        required_keys = ['max_range','max_bearing','std_range','std_bearing', 
                         'pd','clutter_rate']
        if not all(k in params for k in required_keys):
            raise RuntimeError('missing sensor parameters! required keys are: '
                                +str(required_keys))   
        else:
            self.params = params            
        
    def compute_measurement(self,pose,feature):
        dx = feature[0,:] - pose[0] 
        dy = feature[1,:] - pose[1]
        range2 = dx**2 + dy**2
        range = sqrt(range2)
        bearing = wrap_angle( arctan2(dy,dx) )
        z = vstack((range,bearing))
        in_range = logical_and( (range <= self.params['max_range']), 
                                (abs(bearing) <= self.params['max_bearing']) )
        return z[:,in_range]
        
    def compute_noisy_measurement(self,pose,feature):
        # true measurements
        z_clean = self.compute_measurement(pose,feature)
        n_z = size(z_clean,1)
        
        # missed detections
        detected = random.rand(n_z) < self.params['pd']
        z_detected = z_clean[:,detected]
        n_detected = size(z_detected,1)
        
        # measurement noise
        range_noise = random.randn(1,n_detected)*self.params['std_range']
        bearing_noise = random.randn(1,n_detected)*self.params['std_bearing']
        z_noisy = z_detected + vstack((range_noise,bearing_noise))
        z_noisy[1,:] = wrap_angle(z_noisy[1,:])
        
        # clutter
        n_clutter = random.poisson(self.params['clutter_rate'])
        range_clutter = random.rand(1,n_clutter)*self.params['max_range']
        bearing_clutter = (random.rand(1,n_clutter)*self.params['max_bearing']*2 
                            - self.params['max_bearing'] )
        z_clutter = vstack((range_clutter,bearing_clutter))
        return hstack((z_noisy,z_clutter))
        
        
        
        
        

