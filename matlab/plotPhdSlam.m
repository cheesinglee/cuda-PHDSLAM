clear_custom
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

make_movie = false ;

%% parse logs
% t = 0 ;
% nSteps = 0 ;
% 
% % read the backup of the config file to see if the features are dynamic or
% % static
% % config_file = fileread([path,filesep,'config.cfg']) ;
% % expr = 'feature_model\s*=\s*(?<dynamic>\d)' ;
% % matches = regexp(config_file,expr,'names') ;
% % is_dynamic = strcmp(matches.dynamic,'1') ;
% 
% disp 'reading log files...'
% listing = dir([path,filesep,'particles0*']) ;
% listing2 = dir([path,filesep,'state_estimate*']) ;
% nSteps = length(listing) + length(listing2) ;
% nParticles = -1 ;
% expectedMeans = cell(nSteps,1) ;
% expectedCovs = cell(nSteps,1) ;
% expectedWeights = cell(nSteps,1) ;
% means_dynamic = cell(nSteps,1) ;
% covs_dynamic = cell(nSteps,1) ;
% weights_dynamic = cell(nSteps,1) ;
% expectedTraj = zeros(6,nSteps) ;
% estimatedCn = cell(nSteps,1) ;
% particlePoses = cell(nSteps,1) ;
% particleWeights = cell(nSteps,1) ;
% particleWeights = [] ;
% % nSteps = 400 ;
% for i = 1:nSteps
%     disp([num2str(i),'/',num2str(nSteps)]) 
%     matfilename = [path,filesep,'particles',num2str(i-1,'%05d'),'.mat'] ;
%     txtfilename = [path,filesep,'state_estimate',num2str(i-1,'%05d'),'.log'] ;
%     if exist(matfilename,'file') 
%     %     disp( particleFilename )
%         load(matfilename) ;
%         
% %         % allocate space for results on first run
% %         if nParticles < 0
% %             nParticles = length(particles.weights) ;
% %             particleWeights = zeros(nSteps, nParticles) ;
% %             particlePoses = zeros( 6, nParticles, nSteps ) ;
% %         end
%         
%         particle_weights = particles.weights ;
%         particle_poses = particles.states ;
%         
%         % get the heaviest particle
%         [w_max,idx_max] = max(particles.weights) ;
%         if isempty(particles.maps_static(2).weights)
%             idx_max = 1 ;
%         end
%         
%         weighted_poses = particle_poses .* repmat(exp(particle_weights)',6,1) ;
%         expectedTraj(:,i) = sum(weighted_poses,2) ;
%         
%         expectedWeights{i} = particles.maps_static(idx_max).weights ;
%         expectedMeans{i} = particles.maps_static(idx_max).means ;
%         expectedCovs{i} = particles.maps_static(idx_max).covs ;
%         
%         weights_dynamic{i} = particles.maps_dynamic(idx_max).weights ;
%         means_dynamic{i} = particles.maps_dynamic(idx_max).means ;
%         covs_dynamic{i} = particles.maps_dynamic(idx_max).covs ;
%         
%         particlePoses{i} = particle_poses ;
%         particleWeights{i} = particle_weights ;
%     elseif exist(txtfilename,'file')
%         fid=fopen(txtfilename) ;
%         
%         pose_line = fgetl(fid) ;
%         map_line = fgetl(fid) ;
%         weights_line = fgetl(fid) ;
%         particles_line = fgetl(fid) ;
% %         cn_line = fgetl(fid) ;
%         
%         pose_cell = textscan(pose_line,'%f %f %f %f %f %f','CollectOutput',true) ;
%         particles_cell = textscan(particles_line,'%f %f %f %f %f %f') ;
%         weights_cell = textscan(weights_line,'%f') ;
% %         cn_cell = textscan(cn_line,'%f') ;
%         if nParticles < 0
%             nParticles = numel(weights_cell{1}) ;
%             particleWeights = zeros(nSteps,nParticles) ;
%             particlePoses = zeros(nSteps,nParticles,6) ;
%         end
%         pose = pose_cell{1}' ;
%         map_means = [] ;
%         map_covs = [] ;
%         map_weights = [] ;
%         if length(map_line) > 0 
%             map_cell = textscan(map_line,'%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f') ;
%             map_weights = map_cell{1} ;
%             n_features = numel(map_weights) ;
%             if is_dynamic
%                 map_means = [ map_cell{2}' ; map_cell{3}' ; map_cell{4}' ; map_cell{5}'] ;
%                 map_covs = zeros(4,4,n_features) ;
%                 map_covs(1,1,:) = reshape(map_cell{6},[1,1,n_features]) ;
%                 map_covs(2,1,:) = reshape(map_cell{7},[1,1,n_features]) ;
%                 map_covs(3,1,:) = reshape(map_cell{8},[1,1,n_features]) ;
%                 map_covs(4,1,:) = reshape(map_cell{9},[1,1,n_features]) ;
%                 map_covs(1,2,:) = reshape(map_cell{10},[1,1,n_features]) ;
%                 map_covs(2,2,:) = reshape(map_cell{11},[1,1,n_features]) ;
%                 map_covs(3,2,:) = reshape(map_cell{12},[1,1,n_features]) ;
%                 map_covs(4,2,:) = reshape(map_cell{13},[1,1,n_features]) ;
%                 map_covs(1,3,:) = reshape(map_cell{14},[1,1,n_features]) ;
%                 map_covs(2,3,:) = reshape(map_cell{15},[1,1,n_features]) ;
%                 map_covs(3,3,:) = reshape(map_cell{16},[1,1,n_features]) ;
%                 map_covs(4,3,:) = reshape(map_cell{17},[1,1,n_features]) ;
%                 map_covs(1,4,:) = reshape(map_cell{18},[1,1,n_features]) ;
%                 map_covs(2,4,:) = reshape(map_cell{19},[1,1,n_features]) ;
%                 map_covs(3,4,:) = reshape(map_cell{20},[1,1,n_features]) ;
%                 map_covs(4,4,:) = reshape(map_cell{21},[1,1,n_features]) ;
%             else
%                 map_means = [ map_cell{2}' ; map_cell{3}' ] ;
%                 map_covs = zeros(2,2,n_features) ;
%                 map_covs(1,1,:) = reshape(map_cell{4},[1,1,n_features]) ;
%                 map_covs(2,1,:) = reshape(map_cell{5},[1,1,n_features]) ;
%                 map_covs(1,2,:) = reshape(map_cell{6},[1,1,n_features]) ;
%                 map_covs(2,2,:) = reshape(map_cell{7},[1,1,n_features]) ;
%             end
%         end
%         particleWeights(i,:) = weights_cell{1} ;
%         particlePoses(i,:,1) = particles_cell{1} ;
%         particlePoses(i,:,2) = particles_cell{2} ;
%         particlePoses(i,:,3) = particles_cell{3} ;
%         particlePoses(i,:,4) = particles_cell{4} ;
%         particlePoses(i,:,5) = particles_cell{5} ;
%         particlePoses(i,:,6) = particles_cell{6} ;
%         expectedTraj(:,i) = pose ;
%         expectedMeans{i} = map_means ;
%         expectedCovs{i} = map_covs ;
%         expectedWeights{i} = map_weights ;
% %         estimatedCn{i} = cn_cell{1} ;
%         fclose(fid) ;
%     end
% end

logs = parseLogs(path) ;
particlePoses = logs.particlePoses ;
particleWeights = logs.particleWeights ;
expectedTraj = logs.expectedTraj ;
expMapStatic = logs.expMapStatic ;
maxMapStatic = logs.maxMapStatic ;
expMapDynamic = logs.expMapDynamic ;
maxMapDynamic = logs.maxMapDynamic ;
nSteps = logs.nSteps ;
hasExpectedMap = ~isempty(expMapStatic{end}.weights) ;

%% compute errors
% disp('Calculating map errors')
% ospaErrorMax = zeros(1,nSteps) ;
% locErrorMax = zeros(1,nSteps) ;
% cnErrorMax = zeros(1,nSteps) ;
% ospaErrorExp = nan(1,nSteps) ;
% locErrorExp = nan(1,nSteps) ;
% cnErrorExp = nan(1,nSteps) ;
% for k = 1:nSteps
%     disp([num2str(k),'/',num2str(nSteps)])
%     [~,xStatic,~] = maxMapStatic{k}.threshold_targets() ;
%     [~,xDynamic,~] = maxMapDynamic{k}.threshold_targets() ;
%     X = [xStatic,xDynamic(1:2,:)] ;
%     Y = [sim.ground_truth{k}.loc,sim.ground_truth_dyn{k}.loc] ;
%     [ospaErrorMax(k),locErrorMax(k),cnErrorMax(k)] = ospa_dist(X,Y,5,1) ;
%     
%     if hasExpectedMap
%         [~,xStatic,~] = expMapStatic{k}.threshold_targets() ;
%         [~,xDynamic,~] = expMapDynamic{k}.threshold_targets() ;
%         X = [xStatic,xDynamic(1:2,:)] ;
%         Y = [sim.ground_truth{k}.loc,sim.ground_truth_dyn{k}.loc] ;
%         [ospaErrorExp(k),locErrorExp(k),cnErrorExp(k)] = ospa_dist(X,Y,5,1) ;
%     end
% end
% figure(3)
% subplot(3,1,1)
% plot(1:nSteps,ospaErrorMax) ;
% hold on
% plot(1:nSteps,ospaErrorExp,'r') ;
% ylabel 'OSPA Error'
% subplot(3,1,2) 
% plot(1:nSteps,locErrorMax) ;
% hold on
% plot(1:nSteps,locErrorExp,'r') ;
% ylabel 'Localization Error'
% subplot(3,1,3)
% plot(1:nSteps,cnErrorMax) ;
% hold on
% plot(1:nSteps,cnErrorExp,'r') ;
% xlabel 'Time step'
% ylabel 'Cardinality Error'

%% plot
nParticles = numel(particleWeights{end}) ;
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
if make_movie
    avi_frames = struct(getframe(gcf)) ;
    avi_frames = repmat(avi_frames,1,ceil(nSteps/draw_rate)) ;
end
frame_counter = 1 ;
pos_error = zeros(1,nSteps/draw_rate) ;
yaw_error = zeros(1,nSteps/draw_rate) ;
for i = 1:draw_rate:nSteps
    weights = exp(particleWeights{i}) ;
    poses = particlePoses{i} ;
    if size(poses,1) == 1
        poses = poses' ;
    end
%     mapWeights = maxMapStatic{i}.weights ;
%     mapMeans = maxMapStatic{i}.means ;
%     mapCovs = maxMapStatic{i}.covs ;
%     nFeatures = size(mapMeans,2) ;
%     weight_sum = sum(mapWeights) ;
%     [sorted, idx] = sort(mapWeights,'descend') ;
%     cn_est = weight_sum ;
%     if weight_sum > numel(idx)
%         weight_sum = numel(idx) ;
%     end
%     idx = idx(1:round(weight_sum)) ;
%     idx = idx(2:end) ;
%     pp = make_cov_ellipses( mapMeans(1:2,idx)', mapCovs(1:2,1:2,idx), N ) ;
% 
%     mapWeights = maxMapDynamic{i}.weights ;
%     mapMeans = maxMapDynamic{i}.means ;
%     mapCovs = maxMapDynamic{i}.covs ;
%     nFeatures = size(mapMeans,2) ;
%     weight_sum = sum(mapWeights) ;
%     [sorted, idx] = sort(mapWeights,'descend') ;
%     if weight_sum > numel(idx)
%         weight_sum = numel(idx) ;
%     end
%     idx = idx(1:round(weight_sum)) ;
%     idx = mapWeights > min_weight ;
%     pp_dynamic = make_cov_ellipses( mapMeans(1:2,idx)', mapCovs(1:2,1:2,idx), N ) ;

    clf 
    set(0,'CurrentFigure',1) ;
    
    % ------------ MAP map plot --------------- %
    if ~hasExpectedMap
        subplot(2,4,[1,2,5,6])
    else
        subplot(2,4,[1,2])
    end
    hold on
    ppStatic = maxMapStatic{i}.plot_ellipses() ;
    ppDynamic = maxMapDynamic{i}.plot_ellipses() ;
    cn_est = sum(maxMapStatic{i}.weights) + sum(maxMapDynamic{i}.weights) ;
    if ( numel(ppStatic) > 0 )
        plot(ppStatic(1,:),ppStatic(2,:),'b','linewidth',2)
    end
    if( numel(ppDynamic) > 0 )
        plot(ppDynamic(1,:),ppDynamic(2,:),'r','linewidth',2)
    end
    plot(expectedTraj(1,i),expectedTraj(2,i),'dr','Markersize',8) ;
    plot( expectedTraj(1,1:i), expectedTraj(2,1:i), 'r--' ) ;
    plot(poses(1,:),poses(2,:),'.') ;
    if (exist('sim','var'))
        plot( sim.traj(1,:), sim.traj(2,:), 'k' )
        plot( sim.ground_truth{i}.loc(1,:), sim.ground_truth{i}.loc(2,:),'k*')
        if ~isempty(sim.ground_truth_dyn{i}.loc)
            plot( sim.ground_truth_dyn{i}.loc(1,:), sim.ground_truth_dyn{i}.loc(2,:),'g*')
        end
    end
    
    % plot measurements
    if (exist('sim','var'))
        Zi = [sim.z_noisy_static{i},sim.z_noisy_dynamic{i}] ;
        r = Zi(1,:) ;
        b = Zi(2,:) + sim.traj(3,i) ;
        dx = r.*cos(b) ;
        dy = r.*sin(b) ;
        x = sim.traj(1,i) + dx ;
        y = sim.traj(2,i) + dy ;
        z_lines = zeros(2,numel(x)*2) ;
        z_lines(:,1:2:end) = [x;y] ;
        z_lines(:,2:2:end) = repmat(sim.traj(1:2,i),[1,numel(x)]) ;
        plot(z_lines(1,:),z_lines(2,:),'g-.') 
    end
    
    title(num2str(i))
    axis equal
    xlim([-25 25])
    ylim([-30 30])
    grid on
    
    % --------- EAP Map plot ------------------ %
    if hasExpectedMap
        subplot(2,4,[5,6])
        hold on
        ppStatic = expMapStatic{i}.plot_ellipses() ;
        ppDynamic = expMapDynamic{i}.plot_ellipses() ;
        if ( numel(ppStatic) > 0 )
            plot(ppStatic(1,:),ppStatic(2,:),'b','linewidth',2)
        end
        if( numel(ppDynamic) > 0 )
            plot(ppDynamic(1,:),ppDynamic(2,:),'r','linewidth',2)
        end
        plot(expectedTraj(1,i),expectedTraj(2,i),'dr','Markersize',8) ;
        plot( expectedTraj(1,1:i), expectedTraj(2,1:i), 'r--' ) ;
        plot(poses(1,:),poses(2,:),'.') ;
        if (exist('sim','var'))
            plot( sim.traj(1,:), sim.traj(2,:), 'k' )
            plot( sim.ground_truth{i}.loc(1,:), sim.ground_truth{i}.loc(2,:),'k*')
            if ~isempty(sim.ground_truth_dyn{i}.loc)
                plot( sim.ground_truth_dyn{i}.loc(1,:), sim.ground_truth_dyn{i}.loc(2,:),'g*')
            end
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
            plot(z_lines(1,:),z_lines(2,:),'g-.') 
        end

        title(num2str(i))
        axis equal
        xlim([-25 25])
        ylim([-30 30])
        grid on
    end
    
    % --------- particle weights plot ---------- %
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
%     cn = estimatedCn{i} ;
%     plot(0:numel(cn)-1,exp(cn),'.') ;
%     ylim([0,1]) ;
    pos_error_i = norm(expectedTraj(1:2,i)-sim.traj(1:2,i)) ;
    yaw_error_i = wrapAngle(expectedTraj(3,i)-sim.traj(3,i)) ;
    pos_error(i/draw_rate) = pos_error_i ;
    yaw_error(i/draw_rate) = yaw_error_i ;
    plot(1:draw_rate:i,pos_error(1:i/draw_rate),'linewidth',2) ;
    hold on
    plot(1:draw_rate:i,yaw_error(1:i/draw_rate),'r','linewidth',2) ;
    xlim([0,nSteps])
    ylim([0,5])
    ylabel('Vehicle Error')
    xlabel('Time step')
    title(['Cardinality, weightsum = ',num2str(cn_est)])
%     pause
    drawnow
    if make_movie
        avi_frames(frame_counter) = getframe(gcf) ;
    end
    frame_counter = frame_counter + 1 ;
%     keyboard
end

%%
if make_movie
    disp('Creating AVI')
    avi = avifile('scphd_mixed_targets_mixed_models.avi') ;
    for i = 1:numel(avi_frames)
        disp([num2str(i),'/',num2str(numel(avi_frames))])
        avi = addframe( avi,avi_frames(i) ) ;
    end
    avi = close(avi) ;
end