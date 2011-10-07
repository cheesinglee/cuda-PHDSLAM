# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 17:19:48 2011

@author: -
"""

from numpy import *
import glob
from os.path import join
import tables
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.gridspec import GridSpec
from PyQt4.QtCore import *
from PyQt4.QtGui import *
        
class Gaussian2D:
    def __init__(self,weight,mean,cov):
        self.mean = mean
        self.cov = cov
        self.weight = weight
    def draw(self,N):
        w,v = linalg.eig(self.cov)
        phi = linspace(0,2*pi,num=N)
        unit_x = cos(phi)
        unit_y = sin(phi)
        circ = vstack((unit_x,unit_y))
        w = diag(sqrt(w))
        points = dot( dot( circ.transpose(), w ) , 3*v.transpose() )
        points += self.mean
#        line = Line2D(points[:,0],points[:,1],animated=True )
        return points
        
class Plotter(QMainWindow):
    def __init__(self,groundtruthfile,n_particles,parent=None):
        # init parent class
        QMainWindow.__init__(self,parent)
        # load ground truth
        f = tables.openFile(groundtruthfile)
        self.true_traj = f.root.traj[:]
        self.true_map = f.root.staticMap[:]
        f.close()
        
        self.n_particles = n_particles
        self.create_widgets()
        self.logfiles = []
        self.frame = 0 
        self.data_loaded = False
        self.poses = []
        self.maps = []
        self.particle_poses = []
        self.weights = []

        
    def create_widgets(self):
        self.top_widget = QWidget()        
        
        self.fig = Figure(figsize=(12,6))
        self.canvas = FigureCanvas(self.fig)
        
        gs = GridSpec(2,4)        
        
        # main axes
        self.ax = self.fig.add_subplot(gs[:,0:2])
        self.ax.plot( self.true_traj[:,0], self.true_traj[:,1],'k' )
        self.ax.plot( self.true_map[:,0], self.true_map[:,1],'k*')
        
        # particle scatterplot axes
        self.ax_particles = self.fig.add_subplot(gs[0,2])
        self.ax_particles.set_xmargin(0.1)
        self.ax_particles.set_ymargin(0.1)
        
        # particle weights axes
        self.ax_weights = self.fig.add_subplot(gs[0,3])
            
        # cardinality axes
        self.ax_cn = self.fig.add_subplot(gs[1,2:])
        
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.background_particles = self.canvas.copy_from_bbox(self.ax_particles.bbox)
        
        # define animated artists
        init_x = [self.true_traj[0,0]]
        init_y = [self.true_traj[0,1]]
        self.est_pose, = self.ax.plot(init_x, init_y, 'rd', ms=8, mec='r', 
                                      mew=2,mfc='None',animated=True )
        self.est_traj, = self.ax.plot(init_x, init_y, 
                                      'r--',animated=True)   
        self.est_map = []
        self.zlines = []
        for i in xrange(100):
            l, = self.ax.plot([0],[0],color='b',linestyle='None',animated=True)
            self.est_map.append(l)
            l, = self.ax.plot([0],[0],color='g',linestyle='None',animated=True)
            self.zlines.append(l)
        self.particles, = self.ax.plot(init_x*ones(self.n_particles),
                                       init_y*ones(self.n_particles),color='b',
                                       animated=True,marker=',',ls='None')                                     
        self.particles2, = self.ax_particles.plot(init_x*ones(self.n_particles),
                                                  init_y*ones(self.n_particles),
                                                  color='b',animated=True,
                                                  marker='.',ls='None') 
                                                  
        self.load_button = QPushButton('Load Data')
        self.play_button = QPushButton('Play')
        self.play_button.setEnabled(False)
        self.play_button.setCheckable(True)
        self.timer = QTimer()
        self.timer.setInterval(100)
        
        # signals and slots
        self.connect(self.load_button,SIGNAL('clicked()'),self.load_callback)    
        self.connect(self.play_button,SIGNAL('clicked()'),self.play_callback)
        self.connect(self.timer,SIGNAL('timeout()'),self.timer_callback)
        
        hbox = QHBoxLayout()
        hbox.addWidget(self.load_button)
        hbox.addWidget(self.play_button)
        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)
        vbox.addLayout(hbox)
        self.top_widget.setLayout(vbox)                               
        self.setCentralWidget(self.top_widget)                                       
        
    def update_plot(self):
        if not self.data_loaded:
            return
            
        if self.frame >= self.n_steps:
            return
        
        # append the pose to the estimated trajectory
        pose = self.poses[self.frame,:]
        traj_x = self.poses[0:self.frame,0]
        traj_y = self.poses[0:self.frame,1]
        self.est_traj.set_xdata(traj_x)
        self.est_traj.set_ydata(traj_y)
        self.est_pose.set_xdata(pose[0])
        self.est_pose.set_ydata(pose[1])
        
        # compute feature ellipses
        features = self.maps[self.frame]
        ellipses = [ g.draw(10) for g in features ]
        
        # restore the background
        self.canvas.restore_region(self.background)
        self.canvas.restore_region(self.background_particles)

        # draw the animated elements
        particles = self.particle_poses[self.frame]
        self.ax.draw_artist(self.est_traj)
        self.ax.draw_artist(self.est_pose)
        n_features = len(ellipses)
        for i in xrange(n_features):
            l = self.est_map[i]
            e = ellipses[i]
            l.set_xdata(e[:,0])
            l.set_ydata(e[:,1])
            l.set_linestyle('-')
            self.ax.draw_artist(l)
        self.particles.set_xdata(particles[0,:])
        self.particles.set_ydata(particles[1,:])
        self.ax.draw_artist(self.particles)
        self.canvas.blit(self.ax.bbox)
        
        self.particles2.set_xdata(particles[0,:])
        self.particles2.set_ydata(particles[1,:])
        self.ax_particles.draw_artist(self.particles2)
        self.ax_particles.set_xlim( left = min(particles[0,:]), right = max(particles[0,:]))
        self.ax_particles.set_ylim( bottom = min(particles[1,:]), top = max(particles[1,:]))
        self.canvas.blit(self.ax_particles.bbox)
        self.frame += 1
        
    def keyPressEvent(self,evt):
        QMainWindow.keyPressEvent()
        print 'key pressed: ',evt.key
        if evt.key == Qt.Key_Escape:
            self.close()
            
    def resizeEvent(self,evt):
        self.ax.clear()
        self.ax.plot( self.true_traj[:,0], self.true_traj[:,1],'k' )
        self.ax.plot( self.true_map[:,0], self.true_map[:,1],'k*')
        self.ax_particles.clear()
        self.ax_cn.clear()
        self.ax_weights.clear()
        self.canvas.draw()
        
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.background_particles = self.canvas.copy_from_bbox(self.ax_particles.bbox)
        
    def load_data(self):
        data_dir = QFileDialog.getExistingDirectory(plot,
                                                    'Select data directory')
        pattern = join( str(data_dir),'state_estimate*.log' )
        self.logfiles = glob.glob( pattern )
        self.logfiles.sort()
        
        self.n_steps = len(self.logfiles)
        self.frame = 0 
        self.poses = empty([self.n_steps,2])
        self.maps = []
        self.weights = []
        self.particle_poses = []
        for k in xrange(self.n_steps):
            print k
            f = open(self.logfiles[k])
            pose = fromstring( f.readline(), sep=' ' )
            map_all = fromstring( f.readline(), sep=' ' )
            particle_weights = fromstring( f.readline(), sep=' ' )
            particles = fromstring( f.readline(), sep=' ' )
            particles = vstack((particles[0::6],particles[1::6]))
            f.close()
            
            n_features = map_all.size/7
            est_map = []
            for i in xrange(n_features):
                ii = i*7
                weight = map_all[ii]
                if weight >= 0.33:
                    mean = array([map_all[ii+1],map_all[ii+2]])
                    
                    # cov matrix stored in col major format
                    cov = array([[map_all[ii+3], map_all[ii+5]],
                                 [map_all[ii+4], map_all[ii+6]]])
                    est_map.append( Gaussian2D(weight,mean,cov) )
            self.poses[k,:] = pose[0:2]
            self.weights.append(particle_weights)
            self.particle_poses.append(particles)
            self.maps.append(est_map) 
        self.data_loaded = True
        self.play_button.setEnabled(True)
        self.frame = 0
    
    def load_callback(self):
        self.load_data()
    
    def play_callback(self):
        if not self.timer.isActive():
            self.timer.start() 
            self.play_button.setChecked(True)
        else:
            self.timer.stop()
            self.play_button.setChecked(False)
        
    def timer_callback(self):
        self.update_plot()


if __name__ == "__main__":
    app = QApplication([])
    plot = Plotter('groundtruth.mat',256)
    plot.show()   
    app.exec_()     

    
 