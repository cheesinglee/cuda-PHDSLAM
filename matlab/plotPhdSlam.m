clear classes
close all
pause on
path = uigetdir('./','Choose logs directory') ;
if ( path == 0 )
    return 
end

[fname,pname] = uigetfile('../data/*.mat','Choose simulation data file') ;
if ( fname ~= 0 )
    simdata_filename = [pname,filesep,fname] ;
    load(simdata_filename)
end

%% parse logs
t = 0 ;
nSteps = 0 ;

% read the backup of the config file to see if the features are dynamic or
% static
config_file = fileread([path,filesep,'config.cfg']) ;
expr = 'dynamic_features\s*=\s*(?<dynamic>\d)' ;
matches = regexp(config_file,expr,'names') ;
is_dynamic = strcmp(matches.dynamic,'1') ;

disp 'reading log files...'
listing = dir([path,filesep,'particles*']) ;
listing2 = dir([path,filesep,'state_estimate*']) ;
nSteps = length(listing) + length(listing2) ;
nParticles = -1 ;
expectedMeans = cell(nSteps,1) ;
expectedCovs = cell(nSteps,1) ;
expectedWeights = cell(nSteps,1) ;
means_dynamic = cell(nSteps,1) ;
covs_dynamic = cell(nSteps,1) ;
weights_dynamic = cell(nSteps,1) ;
expectedTraj = zeros(6,nSteps) ;
estimatedCn = cell(nSteps,1) ;
particleWeights = [] ;
% nSteps = 400 ;
for i = 1:nSteps
    disp([num2str(i),'/',num2str(nSteps)]) 
    matfilename = [path,filesep,'particles',num2str(i-1,'%05d'),'.mat'] ;
    txtfilename = [path,filesep,'state_estimate',num2str(i-1,'%05d'),'.log'] ;
    if exist(matfilename,'file') 
    %     disp( particleFilename )
        load(matfilename) ;
        
        % allocate space for results on first run
        if nParticles < 0
            nParticles = length(particles.weights) ;
            particleWeights = zeros(nSteps, nParticles) ;
            particlePoses = zeros( 6, nParticles, nSteps ) ;
        end
        
        particle_weights = particles.weights ;
        particle_poses = particles.states ;
        
        % get the heaviest particle
        [w_max,idx_max] = max(particles.weights) ;
        
        weighted_poses = particle_poses .* repmat(exp(particle_weights)',6,1) ;
        expectedTraj(:,i) = sum(weighted_poses,2) ;
        
        expectedWeights{i} = particles.maps_static(idx_max).weights ;
        expectedMeans{i} = particles.maps_static(idx_max).means ;
        expectedCovs{i} = particles.maps_static(idx_max).covs ;
        
        weights_dynamic{i} = particles.maps_dynamic(idx_max).weights ;
        means_dynamic{i} = particles.maps_dynamic(idx_max).means ;
        covs_dynamic{i} = particles.maps_dynamic(idx_max).covs ;
        
        particlePoses(:,:,i) = particle_poses ;
        particleWeights(i,:) = particle_weights ;
%         clear 'paricles'
    elseif exist(txtfilename,'file')
        fid=fopen(txtfilename) ;
        
        pose_line = fgetl(fid) ;
        map_line = fgetl(fid) ;
        weights_line = fgetl(fid) ;
        particles_line = fgetl(fid) ;
%         cn_line = fgetl(fid) ;
        
        pose_cell = textscan(pose_line,'%f %f %f %f %f %f','CollectOutput',true) ;
        particles_cell = textscan(particles_line,'%f %f %f %f %f %f') ;
        weights_cell = textscan(weights_line,'%f') ;
%         cn_cell = textscan(cn_line,'%f') ;
        if nParticles < 0
            nParticles = numel(weights_cell{1}) ;
            particleWeights = zeros(nSteps,nParticles) ;
            particlePoses = zeros(nSteps,nParticles,6) ;
        end
        pose = pose_cell{1}' ;
        map_means = [] ;
        map_covs = [] ;
        map_weights = [] ;
        if length(map_line) > 0 
            map_cell = textscan(map_line,'%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f') ;
            map_weights = map_cell{1} ;
            n_features = numel(map_weights) ;
            if is_dynamic
                map_means = [ map_cell{2}' ; map_cell{3}' ; map_cell{4}' ; map_cell{5}'] ;
                map_covs = zeros(4,4,n_features) ;
                map_covs(1,1,:) = reshape(map_cell{6},[1,1,n_features]) ;
                map_covs(2,1,:) = reshape(map_cell{7},[1,1,n_features]) ;
                map_covs(3,1,:) = reshape(map_cell{8},[1,1,n_features]) ;
                map_covs(4,1,:) = reshape(map_cell{9},[1,1,n_features]) ;
                map_covs(1,2,:) = reshape(map_cell{10},[1,1,n_features]) ;
                map_covs(2,2,:) = reshape(map_cell{11},[1,1,n_features]) ;
                map_covs(3,2,:) = reshape(map_cell{12},[1,1,n_features]) ;
                map_covs(4,2,:) = reshape(map_cell{13},[1,1,n_features]) ;
                map_covs(1,3,:) = reshape(map_cell{14},[1,1,n_features]) ;
                map_covs(2,3,:) = reshape(map_cell{15},[1,1,n_features]) ;
                map_covs(3,3,:) = reshape(map_cell{16},[1,1,n_features]) ;
                map_covs(4,3,:) = reshape(map_cell{17},[1,1,n_features]) ;
                map_covs(1,4,:) = reshape(map_cell{18},[1,1,n_features]) ;
                map_covs(2,4,:) = reshape(map_cell{19},[1,1,n_features]) ;
                map_covs(3,4,:) = reshape(map_cell{20},[1,1,n_features]) ;
                map_covs(4,4,:) = reshape(map_cell{21},[1,1,n_features]) ;
            else
                map_means = [ map_cell{2}' ; map_cell{3}' ] ;
                map_covs = zeros(2,2,n_features) ;
                map_covs(1,1,:) = reshape(map_cell{4},[1,1,n_features]) ;
                map_covs(2,1,:) = reshape(map_cell{5},[1,1,n_features]) ;
                map_covs(1,2,:) = reshape(map_cell{6},[1,1,n_features]) ;
                map_covs(2,2,:) = reshape(map_cell{7},[1,1,n_features]) ;
            end
        end
        particleWeights(i,:) = weights_cell{1} ;
        particlePoses(i,:,1) = particles_cell{1} ;
        particlePoses(i,:,2) = particles_cell{2} ;
        particlePoses(i,:,3) = particles_cell{3} ;
        particlePoses(i,:,4) = particles_cell{4} ;
        particlePoses(i,:,5) = particles_cell{5} ;
        particlePoses(i,:,6) = particles_cell{6} ;
        expectedTraj(:,i) = pose ;
        expectedMeans{i} = map_means ;
        expectedCovs{i} = map_covs ;
        expectedWeights{i} = map_weights ;
%         estimatedCn{i} = cn_cell{1} ;
        fclose(fid) ;
    end
end

%% plot
close all


min_weight = 0.25 ;
figure(1)
set(gcf,'Position',[100,100,800,600]) ;
subplot(2,4,[1,2,5,6])
% h_ellipses = plot(0,0,'b') ;
% h_particles = plot(0,0,'.') ;
hold on
grid on
axis equal
N = 10 ;
draw_rate = 1 ;
avi_frames = struct(getframe(gcf)) ;
avi_frames = repmat(avi_frames,1,ceil(nSteps/draw_rate)) ;
frame_counter = 1 ;
for i = 1:draw_rate:nSteps
    set(0,'CurrentFigure',1) ;
    weights = exp(particleWeights(i,:)) ;
    poses = particlePoses(:,:,i) ;
    if size(poses,1) == 1
        poses = poses' ;
    end
    mapWeights = expectedWeights{i} ;
    mapMeans = expectedMeans{i} ;
    mapCovs = expectedCovs{i} ;
    nFeatures = size(mapMeans,2) ;
    weight_sum = sum(mapWeights) ;
    [sorted, idx] = sort(mapWeights,'descend') ;
    cn_est = weight_sum ;
    if weight_sum > numel(idx)
        weight_sum = numel(idx) ;
    end
    idx = idx(1:round(weight_sum)) ;
%     idx = mapWeights > min_weight ;
    pp = make_cov_ellipses( mapMeans(1:2,idx)', mapCovs(1:2,1:2,idx), N ) ;
%     ppGold = makeCovEllipses( expectedMapsGold(i).means,
%     expectedMapsGold(i).covs,N ) ;

    mapWeights = weights_dynamic{i} ;
    mapMeans = means_dynamic{i} ;
    mapCovs = covs_dynamic{i} ;
    nFeatures = size(mapMeans,2) ;
    weight_sum = sum(mapWeights) ;
    cn_est = cn_est + weight_sum ;
    [sorted, idx] = sort(mapWeights,'descend') ;
    if weight_sum > numel(idx)
        weight_sum = numel(idx) ;
    end
    idx = idx(1:round(weight_sum)) ;
%     idx = mapWeights > min_weight ;
    pp_dynamic = make_cov_ellipses( mapMeans(1:2,idx)', mapCovs(1:2,1:2,idx), N ) ;
%     ppGold = makeCovEllipses( expectedMapsGold(i).means,
%     expectedMapsGold(i).covs,N ) ;

    clf 
    subplot(2,4,[1,2,5,6])
    hold on
    if ( numel(pp) > 0 )
        plot(pp(1,:),pp(2,:),'b','linewidth',2)
    end
    if( numel(pp_dynamic) > 0 )
        plot(pp_dynamic(1,:),pp_dynamic(2,:),'r','linewidth',2)
    end
    plot(expectedTraj(1,i),expectedTraj(2,i),'dr','Markersize',8) ;
    plot( expectedTraj(1,1:i), expectedTraj(2,1:i), 'r--' ) ;
    plot(poses(1,:),poses(2,:),'.') ;
%     set(h_particles,'xdata',poses(1,:),'ydata',poses(2,:)) ;
%     set(h_ellipses,'xdata',pp(1,:),'ydata',pp(2,:)) ;
    if (exist('sim','var'))
        plot( sim.traj(1,:), sim.traj(2,:), 'k' )
%         plot( sim.map(1,:),sim.map(2,:),'k*')
%         dynamic_features_i = sim.feature_traj(:,i,:) ;
%         dynamic_features_i = reshape(dynamic_features_i,2,[]) ;
%         plot( dynamic_features_i(1,:),dynamic_features_i(2,:),'r*') ;
        plot( sim.groundTruth{i}.loc(1,:), sim.groundTruth{i}.loc(2,:),'k*')
    end
    
    % plot measurements
    if (exist('sim','var'))
        Zi = sim.z_noised{i} ;
        r = Zi(1,:) ;
        b = Zi(2,:) + sim.traj(3,i) ;
        dx = r.*cos(b) ;
        dy = r.*sin(b) ;
        x = sim.traj(1,i) + dx ;
        y = sim.traj(2,i) + dy ;
        z_lines = zeros(2,numel(x)*2) ;
        z_lines(:,1:2:end) = [x;y] ;
        z_lines(:,2:2:end) = repmat(sim.traj(1:2,i),[1,numel(x)]) ;
%         Zki = sim.data(i).measurementsClutter ;
%         r = Zki(1,:) ;
%         b = Zki(2,:) + sim.traj(3,i) ;
%         dx = r.*cos(b) ;
%         dy = r.*sin(b) ;
%         x = sim.traj(1,i) + dx ;
%         y = sim.traj(2,i) + dy ;
%         z_lines_clutter = zeros(2,numel(x)*2) ;
%         z_lines_clutter(:,1:2:end) = [x;y] ;
%         z_lines_clutter(:,2:2:end) = repmat(sim.traj(1:2,i),[1,numel(x)]) ;
        plot(z_lines(1,:),z_lines(2,:),'g-.') 
%         plot(z_lines_clutter(1,:),z_lines_clutter(2,:),'r-.') 
    end
    
    title(num2str(i))
    axis equal
    xlim([-25 25])
    ylim([-30 30])
    grid on
    subplot(2,4,3)
    hold on
    cmap = colormap ;
    color_idx = ceil(weights/max(weights)*64) ;
    for j = 1:64
        plot(poses(1,(color_idx==j)),poses(2,(color_idx==j)),'.','Color',cmap(j,:),'MarkerSize',8) ;
    end
    if ( exist('sim','var') )
        plot(sim.traj(1,i),sim.traj(2,i),'pk','MarkerSize',12)
    end
    xrange = max(poses(1,:)) - min(poses(1,:)) ;
    yrange = max(poses(2,:)) - min(poses(2,:)) ;
    if yrange == 0
        yrange = 0.1 ;
    end
    if xrange == 0 
        xrange = 0.1 ;
    end
    axis equal 
    grid on
    title('Particle Weights') 
    
    subplot(2,4,4)
    n_eff = 1/sum(weights.^2)/nParticles ;
    bar(weights,'EdgeColor','none') ;
    ylim([0,5/nParticles])
    title(num2str(n_eff))
    
    subplot(2,4,[7,8])
    cn = estimatedCn{i} ;
    plot(0:numel(cn)-1,exp(cn),'.') ;
    ylim([0,1]) ;
    title(['Cardinality, weightsum = ',num2str(cn_est)])
    drawnow
    avi_frames(frame_counter) = getframe(gcf) ;
    frame_counter = frame_counter + 1 ;
%     keyboard
end
%%
disp('Creating AVI')
avi = avifile('scphd_mixed_targets_mixed_models.avi') ;
for i = 1:numel(avi_frames)
    disp([num2str(i),'/',num2str(numel(avi_frames))])
    avi = addframe( avi,avi_frames(i) ) ;
end
avi = close(avi) ;