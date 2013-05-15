%%
close all
pause on
parentpath = uigetdir('../','Choose parent directory') ;
if ( path == 0 )
    return 
end

listing = rdir([parentpath,filesep,'**',filesep,'loopTime.log']) ;
nRuns = size(listing) ;

%%
for n = 1:nRuns
    idx = find(listing(n).name==filesep,1,'last') ;
    subfolder = listing(n).name(1:idx) ;
    disp(subfolder)
    
    % parse logs
    t = 0 ;
    nSteps = 0 ;

    disp 'reading log files...'
    listing = dir([path,filesep,'particles0*']) ;
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
    particlePoses = cell(nSteps,1) ;
    particleWeights = cell(nSteps,1) ;
    particleWeights = [] ;
    % nSteps = 400 ;
    for i = 1:nSteps
        disp([num2str(i),'/',num2str(nSteps)]) 
        matfilename = [path,filesep,'particles',num2str(i-1,'%05d'),'.mat'] ;
        txtfilename = [path,filesep,'state_estimate',num2str(i-1,'%05d'),'.log'] ;
        if exist(matfilename,'file') 
            load(matfilename) ;
            particle_weights = particles.weights ;
            particle_poses = particles.states ;

            % get the heaviest particle
            [w_max,idx_max] = max(particles.weights) ;
            if isempty(particles.maps_static(2).weights)
                idx_max = 1 ;
            end

            weighted_poses = particle_poses .* repmat(exp(particle_weights)',6,1) ;
            expectedTraj(:,i) = sum(weighted_poses,2) ;

            expectedWeights{i} = particles.maps_static(idx_max).weights ;
            expectedMeans{i} = particles.maps_static(idx_max).means ;
            expectedCovs{i} = particles.maps_static(idx_max).covs ;

            weights_dynamic{i} = particles.maps_dynamic(idx_max).weights ;
            means_dynamic{i} = particles.maps_dynamic(idx_max).means ;
            covs_dynamic{i} = particles.maps_dynamic(idx_max).covs ;

            particlePoses{i} = particle_poses ;
            particleWeights{i} = particle_weights ;
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
end