# -*- coding: utf-8 -*-

from numpy import *

def wrap_angle(a):
    rem = remainder(a,2*pi)
    if rem > pi:
        rem -= 2*pi
    elif rem < -pi:
        rem += 2*pi
    return rem

class AckermanMotionModel:
    def __init__(self,params):
        # check that all the sensor parameters are present
        required_keys = ['l','h','a','b','std_encoder','std_alpha']
        if not all(k in params for k in required_keys):
            raise RuntimeError('missing sensor parameters! required keys are: '
                                +str(required_keys))   
        else:
            self.params = params
            
    def compute_motion(self,pose,v_encoder,alpha,dt):
        x_prev = pose[0]
        y_prev = pose[1]
        theta_prev = pose[2]
        l = self.params['l']
        h = self.params['h']
        a = self.params['a']
        b = self.params['b']
        vc = v_encoder/(1 - tan(alpha)*h/l)
        xc_dot = vc*cos(theta_prev)
        yc_dot = vc*sin(theta_prev)
        thetac_dot = vc*tan(alpha)/l
        
        x = x_prev + dt*( xc_dot - thetac_dot*( a*sin(theta_prev) + b*cos(theta_prev) ) )
        y = y_prev + dt*( yc_dot + thetac_dot*( a*cos(theta_prev) - b*sin(theta_prev) ) )
        theta = theta_prev + dt*thetac_dot
        theta = wrap_angle(theta)
        new_pose = vstack((x,y,theta))
        return new_pose
        
