# -*- coding: utf-8 -*-

class ConstantVelocityMotionModel:
    def __init__(self,params):
        # check that all the sensor parameters are present
        required_keys = ['std_x','std_y','std_z','std_yaw','std_pitch']
        if not all(k in params for k in required_keys):
            raise RuntimeError('missing sensor parameters! required keys are: '
                                +str(required_keys))   
        else:
            self.params = params
            
    def compute_motion(self,pose,dt):
        x_prev = pose[0]
        y_prev = pose[1]
        z_prev = pose[2]
        yaw_prev = pose[3]
        
        vx = pose[4] 
        vy = pose[5]
        vz = pose[6]
        vyaw = pose[7]
        
        x = x_prev + dt*vx*cos(yaw_prev) - dt*vy*sin(yaw_prev)
        y = y_prev + dt*vx*sin(yaw_prev) + dt*vy*cos(yaw_prev)
        z = z_prev + dt*vz
        yaw = yaw_prev + dt*vyaw
        new_pose = vstack((x,y,z,yaw,vx,vy,vz,vyaw))
        return new_pose
        
    def compute_noisy_motion(self,pose,dt):
        clean = self.compute_motion(pose,dt)
        
