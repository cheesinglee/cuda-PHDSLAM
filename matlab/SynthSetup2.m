global figWidth figHeight

% % add path to slamtools
% addpath('../slamtools')

% simulation data file
simDataFilename = 'simData2.mat' ;
regenSimData = true ;
loadMeasurementsFromText = false ;

chi2 = chi2inv(0.99,1:1000);
nSteps = 1000;         % number of simulation steps
% time units between each step
dt = 1 ;
DrawEveryNFrames =1e0;   % how often shall we draw?
animationDelay = 0.05 ; % delay between frames
pause off % delay on or off
saveAvi = true ;
figWidth = 800 ; 
figHeight = 600 ;
maxLmapSize = 20;       % maximum number of features in local submaps

motion_type = 0 ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Model parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sensor model
SensorSettings.FieldOfView = 180;       % in degrees (-degrees < 0 < +degrees)
SensorSettings.Range = 10;              % in meters
SensorSettings.Rng_noise = 1.0 ;%1.5;         % in meters (std deviation) 1.5
SensorSettings.Brg_noise = deg2rad(2.0) ;%deg2rad(4.0);           % in degrees (std deviation) 1.0
SensorSettings.clutterRate = 20 ; % mean number of clutter measurements per scan
SensorSettings.Pd = 0.95 ;
R = diag([SensorSettings.Rng_noise; SensorSettings.Brg_noise]).^2;
measurementModel = RangeBearingMeasurementModel2(SensorSettings);

% odometry noise (not used, but required to reuse function to generate
% simulation inputs
StdOdometryNoise=[.1; .1; deg2rad(1)];    % in metres & degrees

initial_state = [0,-15,0,0.2,0,0]' ;

if motion_type == 0
    % CV motion model process noise (acceleration)
    StdAccNoiseX=0.005;                                        % X axis acceleration noise. No sideslip.
    StdAccNoiseY= 0.005;                                         % Y axis acceleration noise
    StdAccNoiseYaw = deg2rad(0.1) ;
    StdAccNoise=[StdAccNoiseX; StdAccNoiseY; StdAccNoiseYaw] ;
    Q = diag(StdAccNoise).^2;
    % modeling parameters
    motion_model = ConstantVelocity2DMotionModel(Q);
elseif motion_type == 1
    % victoria park motion model
    vehicleParams.l = 2.83;
    vehicleParams.h = 0.76;
    vehicleParams.a = 3.78;
    vehicleParams.b = 1.21-1.42/2; 
    sigmaVel = 2; %2
    sigmaSteer = deg2rad(5); %10
    Q = diag([sigmaVel,sigmaSteer]).^2;
    motion_model = AckermanMotionModel(vehicleParams,Q);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Static Map definition
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mapArea = [[-20,-20];[20,20]] ;
mapRange = diff(mapArea) ;

% grid map
% [mapx,mapy] = meshgrid(1:10:50);
% staticMap = [mapx(:)' ; mapy(:)']+10;

% random map
nFeatures = 50 ;
staticMap = [   rand(1,nFeatures)*mapRange(1)+mapArea(1,1) ; 
                rand(1,nFeatures)*mapRange(2)+mapArea(1,2) ] ;

% sort by distance to origin
dists = sum(staticMap.^2) ;
[aaa,idx] = sort(dists,'ascend') ;
staticMap = staticMap(:,idx) ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% make dynamic features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dynamicFeatures = cell(1,nSteps) ;
% dfTraj = generateCircleTrajectory(staticMap, nSteps,2) ;
% dynamicFeatures = generateDynamicFeatures(1, nSteps, dfTraj ) ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Vehicle trajectory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if motion_type == 0
    traj = generateCVTrajectory(initial_state,Q,nSteps,dt) ;
    control = zeros(2,nSteps) ;
elseif motion_type == 1
    [traj,control] = generateAckermanTrajectory(initial_state, motion_model, staticMap ) ;
end
nSteps = size(traj,2) ;
control_tmp.dt = 10*dt ;
W = motion_model.computeControlJacobian(initial_state,control_tmp) ;
initialCovariance = W*Q*W' ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load or generate simulation data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sim_map = staticMap ;
if (~exist(simDataFilename,'file') || regenSimData)
    disp('Generating vehicle trajectory...')
else
    disp('Loading pre-existing trajectory...')
    load(simDataFilename) ;
    traj = sim.traj ;
    sim_map = sim.map ;
end
disp('Generating simulation inputs...')
sim = generateSimData(sim_map,dynamicFeatures,traj,measurementModel,StdOdometryNoise,simDataFilename) ;
sim.control = control ;
if(loadMeasurementsFromText)
    disp('Loading measurements from text file...')
    fid = fopen('measurements.txt') ;
    line = fgetl(fid) ;
    for k = 1:nSteps 
        line = fgetl(fid) ;
        C = textscan(line,'%f %f') ;
        sim.data(k).measurements = [C{1}, C{2}]' ;
    end
    fclose(fid) ;
end
disp( 'Exporting measurements...')
generateMeasurementTextFile(simDataFilename) ;
save(simDataFilename,'sim') ;
disp('Done!')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% preview the trajectory and map
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(77)
clf
plot(traj(1,:),traj(2,:))
hold on
plot(staticMap(1,:),staticMap(2,:),'k*')
grid on
axis equal
title 'Preview'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Specify data association
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dataAssociation = JcbbDataAssociation;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% specify the SLAM algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
slam = {} ;
% EKF SLAM
% slam = [slam {EkfSLAM(motionModel,measurementModel,dataAssociation,traj(:,1))}] ;

% FastSlam 1.0
% slam = [slam {FastSlam1(motionModel,measurementModel,dataAssociation,'InitialPosition',traj(:,1), 'nParticles',50,'resampleThreshold',0.8,'InflateNoise',1)}] ;

% Rao-Blackwellized PHD SLAM
% slam = [slam {RbPhdSlam(motionModel,measurementModel,'InitialPosition',initialState,'InitialCovariance',initialCovariance,'nParticles',100,'resampleThreshold' ,0.8,'InflateNoise',1,'WeightScheme','cluster')}] ;
% slam = [slam {RbPhdSlam(motionModel,measurementModel,'InitialPosition',traj(:,1),'nParticles',50,'resampleThreshold' ,0.5,'InflateNoise',1,'WeightScheme','emptymap')}] ;
% slam = [slam {RbPhdSlam(motionModel,measurementModel,'InitialPosition',traj(:,1),'nParticles',50,'resampleThreshold' ,0.5,'InflateNoise',1,'WeightScheme','singlefeature')}] ;


% Rao-Blackwellized PHD SLAM with Jump Markov Model
% featureProcessNoise = diag([1,1]) ;
% targetMotionModels = [ConstantPosition2DMotionModel(),ConstantVelocity2DMotionModel(featureProcessNoise)] ;
% 
% jmmMeasurementModel = RangeBearingMeasurementModel2(SensorSettings); 
% slam{1} = JumpMarkovRbPhdSlam(motionModel,jmmMeasurementModel,targetMotionModels,'InitialPosition',traj(1:3,1),'nParticles',5,'resampleThreshold',0.3) ;

% Rao-Blackwellized CPHD SLAM
slam = [slam {RbCphdSlam(motion_model,measurementModel,'InitialPosition',initial_state,'InitialCovariance',initialCovariance,'nParticles',100,'resampleThreshold' ,0.8,'InflateNoise',1,'WeightScheme','cluster')}] ;
% slam = [slam {RbCphdSlam(motionModel,measurementModel,'InitialPosition',traj(1:3,1),'nParticles',50,'resampleThreshold' ,0.8,'InflateNoise',1,'WeightScheme','emptymap')}] ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%5
% Initialize log
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
log.time = zeros(1,nSteps) ;
log.nFeatures = zeros(1,nSteps) ;
log.mapOspaErr = zeros(3,nSteps) ;
log.poseErr = zeros(motion_model.stateDOF,nSteps) ;
log.poseCov = zeros(motion_model.stateDOF,motion_model.stateDOF,nSteps) ;
log.Z = cell(1,nSteps) ;
log.groundTruth = cell(1,nSteps) ;
log.slamOut = cell(1,nSteps) ;
% log.crlb = zeros(3,3,nSteps) ;
%log.control = [] ;
%log.slam = [] ;
log.time(1) = 0 ;
log.nFeatures(1) = 0 ;
log.mapOspaErr(:,1) = zeros(3,1) ;
log.poseErr(:,1) = 0 ;
log.poseCov(:,:,1) = zeros(motion_model.stateDOF) ;
log.simData = [] ;
log = repmat(log,1,length(slam)) ;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
% Prepare avi files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
avidir = '/home/cheesinglee/matlab_work/experiments/' ;
if (saveAvi)
    nAlgorithms = length(slam) ;
    avi = cell(1,nAlgorithms) ;
    filenames = {} ;
    for n = 1:nAlgorithms
        avifilename = [avidir,class(slam{n})] ;
        i = 1 ;
        while ( any(strcmp(avifilename,filenames) ) )
            avifilename = [avidir,class(slam{n}),num2str(i)] ;
            i = i + 1 ;
        end
        filenames = [filenames,avifilename] ;
        avi{n} = avifile(avifilename) ;
    end
end
 
