close all
pause on
path = uigetdir('./','Choose logs directory') ;
if ( path == 0 )
    return 
end

[fname,pname] = uigetfile('*.mat','Choose simulation data file') ;
simdata_filename = [pname,filesep,fname] ;

%% parse logs
t = 0 ;
nSteps = 0 ;

disp 'reading log files...'
listing = dir([path,filesep,'state_estimate*']) ;
nSteps = length(listing) ;
nParticles = -1 ;
expectedMeans = cell(nSteps,1) ;
expectedCovs = cell(nSteps,1) ;
expectedWeights = cell(nSteps,1) ;
expectedTraj = zeros(6,nSteps) ;
estimatedCn = cell(nSteps,1) ;
for i = 1:nSteps
            disp([num2str(i),'/',num2str(nSteps)]) 
    matfilename = [path,filesep,'state_estimate',num2str(i-1),'.mat'] ;
    txtfilename = [path,filesep,'state_estimate',num2str(i-1),'.log'] ;
    if exist(matfilename,'file') 
    %     disp( particleFilename )
        load(matfilename) ;
        if nParticles < 0
            nParticles = length(expectation.particles.weights) ;
            particleWeights = zeros(nSteps, nParticles) ;
            particlePoses = zeros( nSteps, nParticles, 6 ) ;
    %         particleMaps = cell(nSteps,nParticles) ;
        end
        particleWeights(i,:) = expectation.particles.weights ;
        statesReshaped = reshape(expectation.particles.poses,[1,6,nParticles]) ;
        particlePoses(i,:,:) = permute(statesReshaped, [1,3,2]) ;
    %     for j = 1:length(particles.maps)
    %         particleMaps{i,j} = particles.maps(j) ;
    %     end
        expectedTraj(:,i) = expectation.pose ;
        expectedMeans{i} = expectation.map.means ;
        expectedCovs{i} = expectation.map.covs ;
        expectedWeights{i} = expectation.map.weights ;
        estimatedCn{i} = exp(expectation.cardinality) ;
        clear 'expectation'
    elseif exist(txtfilename,'file')
        fid=fopen(txtfilename) ;
        
        pose_line = fgetl(fid) ;
        map_line = fgetl(fid) ;
        weights_line = fgetl(fid) ;
        particles_line = fgetl(fid) ;
        
        pose_cell = textscan(pose_line,'%f %f %f %f %f %f','CollectOutput',true) ;
        particles_cell = textscan(particles_line,'%f %f %f %f %f %f') ;
        weights_cell = textscan(weights_line,'%f') ;
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
            map_cell = textscan(map_line,'%f %f %f %f %f %f %f') ;
            map_weights = map_cell{1} ;
            n_features = numel(map_weights) ;
            map_means = [ map_cell{2}' ; map_cell{3}' ] ;
            map_covs = zeros(2,2,n_features) ;
            map_covs(1,1,:) = reshape(map_cell{4},[1,1,n_features]) ;
            map_covs(2,1,:) = reshape(map_cell{5},[1,1,n_features]) ;
            map_covs(1,2,:) = reshape(map_cell{6},[1,1,n_features]) ;
            map_covs(2,2,:) = reshape(map_cell{7},[1,1,n_features]) ;
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
        fclose(fid) ;
    end
end

% %% recompute expected map from individual particles 
% disp 'Computing expected maps...'
% minWeight = 1e-5 ;
% minSeparation = 10 ;
% maxGaussians = 100 ;
% minExpectedWeight = 0.33 ;
% expectedMapsGold = struct('weights',[],'means',[],'covs',[]) ;
% expectedMapsGold = repmat(expectedMapsGold,1,nSteps) ;
% for i = 1:nSteps
%     expectedMapsGold(i) = computeExpectedMap( {particleMaps{i,:}}, ...
%         particleWeights(i,:), minWeight, minSeparation, maxGaussians,...
%         minExpectedWeight ) ;
% end


%% plot
avi = avifile('vo_1feature.avi') ;
min_weight = 0.33 ;
figure(1)
set(gcf,'Position',[100,100,1280,720]) ;
load(simdata_filename)
trajTrue = sim.traj ;
N = 10 ;
for i = 1:nSteps-1
    set(0,'CurrentFigure',1) ;
    weights = exp(particleWeights(i,:)) ;
    poses = squeeze(particlePoses(i,:,:))' ;
    if size(poses,1) == 1
        poses = poses' ;
    end
    mapWeights = expectedWeights{i} ;
    mapMeans = expectedMeans{i} ;
    mapCovs = expectedCovs{i} ;
    nFeatures = size(mapMeans,2) ;
    weight_idx = mapWeights > min_weight ;
    weight_sum = sum(mapWeights) ;
    [sorted, idx] = sort(mapWeights,'descend') ;
    idx = idx(1:round(weight_sum)) ;
    pp = makeCovEllipses( mapMeans(:,idx), mapCovs(:,:,idx), N ) ;
%     ppGold = makeCovEllipses( expectedMapsGold(i).means, expectedMapsGold(i).covs,N ) ;
    clf 
    subplot(2,4,[1,2,5,6])
    hold on
    if ( numel(pp) > 0 )
        plot(pp(1,:),pp(2,:),'b')
    end
    plot(expectedTraj(1,i),expectedTraj(2,i),'dr','Markersize',8) ;
    plot( expectedTraj(1,1:i), expectedTraj(2,1:i), 'r--' ) ;
    plot(poses(1,:),poses(2,:),'.') ;
    plot( trajTrue(1,:), trajTrue(2,:), 'k' )
    plot( sim.groundTruth{i}.loc(1,:), sim.groundTruth{i}.loc(2,:),'k*')
    title(num2str(i))
%     xlim([-10 60])
%     ylim([-10 60])
    axis equal
    grid on
    subplot(2,4,3)
    hold on
    cmap = colormap ;
    color_idx = ceil(weights/max(weights)*64) ;
    for j = 1:64
        plot(poses(1,(color_idx==j)),poses(2,(color_idx==j)),'.','Color',cmap(j,:),'MarkerSize',8) ;
    end
    plot(trajTrue(1,i),trajTrue(2,i),'pk','MarkerSize',12)
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
    bar(weights,'EdgeColor','none') ;
    ylim([0,5/nParticles])
    subplot(2,4,[7,8])
    title('Cardinality')
    cn = estimatedCn{i} ;
    plot(0:numel(cn)-1,cn,'.') ;
    ylim([0,1]) ;
    drawnow
    avi = addframe( avi,getframe(gcf) ) ;
%     subplot(1,3,3)
%     hold on
%     plot(ppGold(1,:),ppGold(2,:),'b')
%     plot(expectedTraj(1,i),expectedTraj(2,i),'dr','Markersize',8) ;
%     plot( expectedTraj(1,1:i), expectedTraj(2,1:i), 'r--' ) ;
%     plot(poses(1,:),poses(2,:),'.') ;
%     plot( trajTrue(1,:), trajTrue(2,:), 'k' )
%     plot( sim.groundTruth{i}.loc(1,:), sim.groundTruth{i}.loc(2,:),'k*')
%     title(num2str(i))
%     xlim([-10 60])
%     ylim([-10 60])
%     axis square
%     drawnow 
    pause(0.02) ;
end
avi = close(avi) ;