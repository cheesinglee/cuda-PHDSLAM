//#include "slamtypes.h"
//#include "cutil.h"
//#include <algorithm>

///// determine which features are in range
///*!
//  * Each thread block handles a single particle. The threads in the block
//  * evaluate the range and bearing [blockDim] features in parallel, looping
//  * through all of the particle's features.

//    \param predictedFeatures Features from all particles concatenated into a
//        single array
//    \param mapSizes Number of features in each particle, so that the function
//        knows where the boundaries are in predictedFeatures
//    \param nParticles Total number of particles
//    \param poses Array of particle poses
//    \param inRange Pointer to boolean array that is filled by the function.
//        For each feature in predictedFeatures that is in range of its
//        respective particle, the corresponding entry in this array is set to
//        true
//    \param nInRange Pointer to integer array that is filled by the function.
//        Should be allocated to have [nParticles] elements. Each entry
//        represents the number of in range features for each particle.
//  */
//__global__ void
//computeInRangeKernel( Gaussian2D *predictedFeatures, int* map_offsets, int nParticles,
//                ConstantVelocityState* poses, char* inRange, int* nInRange )
//{
//    int tid = threadIdx.x ;

//    // total number of predicted features per block
//    int n_featuresBlock ;
//    // number of inrange features in the particle
//    __shared__ int nInRangeBlock ;
//    // vehicle pose of the thread block
//    ConstantVelocityState blockPose ;

//    Gaussian2D feature ;
//    for ( int p = 0 ; p < nParticles ; p += gridDim.x )
//    {
//        if ( p + blockIdx.x < nParticles )
//        {
//            int predict_offset = 0 ;
//            // compute the indexing offset for this particle
//            int map_idx = p + blockIdx.x ;
//            predict_offset = map_offsets[map_idx] ;
//            // particle-wide values
//            if ( tid == 0 )
//                nInRangeBlock = 0 ;
//            blockPose = poses[map_idx] ;
//            n_featuresBlock = mapSizes[map_idx] ;
//            __syncthreads() ;

//            // loop through features
//            for ( int i = 0 ; i < n_featuresBlock ; i += blockDim.x )
//            {
//                if ( tid+i < n_featuresBlock )
//                {
//                    // index of thread feature
//                    int featureIdx = predict_offset + tid + i ;
//                    feature = predictedFeatures[featureIdx] ;

//                    // default value
//                    inRange[featureIdx] = 0 ;

//                    // compute the predicted measurement
//                    REAL dx = feature.mean[0] - blockPose.px ;
//                    REAL dy = feature.mean[1] - blockPose.py ;
//                    REAL r2 = dx*dx + dy*dy ;
//                    REAL r = sqrt(r2) ;
//                    REAL bearing = wrapAngle(atan2f(dy,dx) - blockPose.ptheta) ;
//                    if ( r < dev_config.maxRange &&
//                         fabs(bearing) < dev_config.maxBearing )
//                    {
//                        atomicAdd( &nInRangeBlock, 1 ) ;
//                        inRange[featureIdx] = 1 ;
//                    }
//                }
//            }
//            // store nInrange
//            __syncthreads() ;
//            if ( tid == 0 )
//            {
//                nInRange[map_idx] = nInRangeBlock ;
//            }
//        }
//    }
//}

//__global__ void
//preUpdateKernel( Gaussian2D* maps, RangeBearingMeasurement Z, int n_measure, int* map_offsets )
//{
//    int feature_idx = threadIdx.x ;
//    int z_idx = threadIdx.y ;
//    int map_idx = blockIdx.x ;
//    int predict_offset = map_offsets[map_idx] ;

//}

//void fastSlamUpdate( ParticleSLAM& particles )
//{
//    int n_particles = particles.nParticles ;
//    int total_features = 0 ;
//    for ( int i = 0 ;i < n_particles ; i++ )
//        total_features += particles.maps[i].size() ;
//    Gaussian2D* maps_concat = (Gaussian2D*)malloc(total_features*sizeof(Gaussian2D)) ;
//    int* map_offsets = (int*)malloc((n_particles+1)*sizeof(int)) ;
//    int offset = 0 ;
//    for ( int i = 0 ; i < n_particles ; i++ )
//    {
//        map_offsets[i] = offset ;
//        std::copy( particles.maps[i].begin(), particles.maps[i].end(),
//                   maps_concat+offset ) ;
//        offset += particles.maps[i].size() ;
//    }
//    map_offsets[n_particles] = offset ;

//    /*
//     * separate in-range and out-of-range features
//     */
//    Gaussian2D* dev_map_concat = NULL ;
//    ConstantVelocityState* dev_poses = NULL ;
//    int* dev_map_offsets = NULL ;
//    char* dev_inrange = NULL ;
//    int* dev_n_inrange = NULL ;

//    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_map_concat,
//                                total_features*sizeof(Gaussian2D) ) ) ;
//    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_poses,
//                                n_particles*sizeof(ConstantVelocityState) ) ) ;
//    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_map_offsets,
//                                (n_particles+1)*sizeof(int) ) ) ;
//    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_inrange,
//                                total_features*sizeof(char) ) ) ;
//    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_n_inrange,
//                                n_particles*sizeof(int) ) ) ;

//}
