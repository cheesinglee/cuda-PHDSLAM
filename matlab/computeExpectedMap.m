function [ expectedMap ] = computeExpectedMap( particleMaps, particleWeights,...
    minWeight, minSeparation, maxGaussians, minExpectedWeight )
%COMPUTEEXPECTEDMAP Summary of this function goes here
%   Detailed explanation goes here
    nMaps = size(particleMaps,2) ;
    concatWeights = [] ;
    concatMeans = [] ;
    concatCovs = [] ;
    for n = 1:nMaps
        map = particleMaps{n} ;
        concatWeights = [concatWeights, particleWeights(n)*map.weights'] ;
        concatMeans = [concatMeans, map.means] ;
        concatCovs = cat(3,concatCovs, map.covs) ;
    end
    if ( length(concatWeights) > 0)
        [wMerge,muMerge,pMerge] = pruneGaussianMixture(concatWeights,concatMeans,...
            concatCovs, minWeight,minSeparation,maxGaussians) ;
        idx = wMerge > minExpectedWeight ;
        expectedMap.weights = wMerge(idx) ;
        expectedMap.means = muMerge(:,idx) ;
        expectedMap.covs = pMerge(:,:,idx) ;
    else
        expectedMap.weights = [] ;
        expectedMap.means = [] ;
        expectedMap.covs = [] ;
    end
   
end

