/*
 * main.cpp
 *
 *  Created on: Mar 24, 2011
 *      Author: cheesinglee
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <string>
#include <cstdarg>
#include "slamtypes.h"
//#include "slamparams.h"
//#include <complex.h>
//#include <fftw3.h>
#include <assert.h>
#include <float.h>
#include "cuPrintf.cu"

#include "device_math.cuh"

#include <helper_cuda.h>
#include <curand_kernel.h>

#include <thrust/version.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/remove.h>
#include <thrust/copy.h>
#include <thrust/partition.h>
#include <thrust/scan.h>
#include <thrust/iterator/zip_iterator.h>

//#include "ConstantVelocity2DKinematicModel.cu"

// include gcc-compiled boost rng
#include "rng.h"

#ifdef __CDT_PARSER__
#define __device__
#define __global__
#define __constant__
#define __shared__
#define __host__
#endif

#define DEBUG

#ifdef DEBUG
#define DEBUG_MSG(x) cout << "[" << __func__ << "(" << __LINE__ << ")]: " << x << endl
#define DEBUG_VAL(x) cout << "[" << __func__ << "(" << __LINE__ << ")]: " << #x << " = " << x << endl
#else
#define DEBUG_MSG(x)
#define DEBUG_VAL(x)
#endif

//--- Make kernel helper functions externally visible
void
initCphdConstants() ;

void
initRandomNumberGenerators() ;

void
predictMap(SynthSLAM& p) ;

void
phdPredict(SynthSLAM& particles, ... ) ;

template<class GaussianType>
void
phdPredictVp( SynthSLAM& particles ) ;

SynthSLAM
phdUpdate(SynthSLAM& particles, measurementSet measurements) ;

template <typename T>
T resampleParticles(T oldParticles, int n_particles=-1 ) ;

void
recoverSlamState(SynthSLAM& particles, ConstantVelocityState& expectedPose,
                 vector<REAL>& cn_estimate ) ;

void
recoverSlamState(DisparitySLAM& particles, ConstantVelocityState& expectedPose ) ;

void
setDeviceConfig( const SlamConfig& config ) ;
//--- End external declarations

//template<class GaussianType>
//__host__ __device__ REAL
//wrapAngle(REAL a) ;
//--- End external function declaration

// SLAM configuration, externally declared
extern SlamConfig config ;

// device memory limit, externally declared
extern size_t deviceMemLimit ;

// dynamic shared memory
extern __shared__ REAL shmem[] ;

using namespace std ;
using namespace thrust ;

// Constant memory variables
__device__ __constant__ RangeBearingMeasurement Z[256] ;
__device__ __constant__ SlamConfig dev_config ;

// RNG state variables
__device__ curandStateMRG32k3a_t* devRNGStates ;

// other global device variables
REAL* dev_C ;
REAL* dev_factorial ;
REAL* log_factorials ;
//__device__ REAL* dev_qspower ;
//__device__ REAL* dev_pspower ;
REAL* dev_cn_clutter ;

//ConstantVelocityModelProps modelProps  = {STDX, STDY,STDTHETA} ;
//ConstantVelocity2DKinematicModel motionModel(modelProps) ;

__global__ void initRNGKernel(){
    int tid = blockIdx.x*blockDim.x + threadIdx.x ;
    curand_init(0,tid,0,&devRNGStates[tid]);
}

void initRandomNumberGenerators(){
    int n = config.nSamples ;
    curandStateMRG32k3a_t* states ;
    checkCudaErrors(cudaMalloc((void**)&states,
                               n*sizeof(curandStateMRG32k3a_t))
                    ) ;
    checkCudaErrors(cudaMemcpyToSymbol(devRNGStates,&states,
                                       sizeof(curandStateMRG32k3a_t*))
                    ) ;
    int nBlocks = ceil(n/512.0) ;
    int nThreads = min(512,n) ;
    DEBUG_MSG("Initializing Random Number Generators") ;
    initRNGKernel<<<nBlocks,nThreads>>>() ;
}

/// helper function for outputting a Gaussian to std_out
template<class GaussianType>
__host__  void
print_feature(GaussianType f)
{
    int dims = getGaussianDim(f) ;
//#if defined(__CUDA_ARCH__)
//#warning __CUDA_ARCH__ is defined
//    cuPrintf("%f ",f.weight) ;
//    for ( int i = 0 ; i < dims ; i++ )
//        cuPrintf("%f ",f.mean[i]) ;
//    for ( int i = 0 ; i < dims*dims ; i++ )
//        cuPrintf("%f ",f.cov[i]) ;
//    cuPrintf("\n") ;
//#else
//#warning __CUDA_ARCH__ is not defined
    cout << f.weight << " " ;
    for ( int i = 0 ; i < dims ; i++ )
        cout << f.mean[i] << " " ;
    for ( int i = 0 ; i < dims*dims ; i++)
        cout << f.cov[i] << " " ;
    cout << endl ;
//#endif
}

/// combine all features from all particles into a single STL vector
template<class GaussianType>
vector<GaussianType> combineFeatures(vector<vector <GaussianType> > maps, ...)
{
    vector<GaussianType> concat ;
    for ( unsigned int n = 0 ; n < maps.size() ; n++ )
        concat.insert( concat.end(), maps[n].begin(), maps[n].end()) ;
    return concat ;
}

/// return the next highest power of two
int nextPowerOfTwo(int a)
{
    int n = a - 1 ;
    n = n | (n >> 1) ;
    n = n | (n >> 2) ;
    n = n | (n >> 4) ;
    n = n | (n >> 8);
    n = n | (n >> 16) ;
    n = n + 1 ;
    return n ;
}

__device__ void
computeBirth( ConstantVelocityState pose, RangeBearingMeasurement z,
              Gaussian2D& feature_birth)
{
    // invert measurement
    REAL theta = pose.ptheta + z.bearing ;
    REAL dx = z.range*cos(theta) ;
    REAL dy = z.range*sin(theta) ;
    feature_birth.mean[0] = pose.px + dx ;
    feature_birth.mean[1] = pose.py + dy ;

    // inverse measurement jacobian
    REAL J[4] ;
    J[0] = dx/z.range ;
    J[1] = dy/z.range ;
    J[2] = -dy ;
    J[3] = dx ;

    // measurement noise
    REAL var_range = pow(dev_config.stdRange*dev_config.birthNoiseFactor,2) ;
    REAL var_bearing = pow(dev_config.stdBearing*dev_config.birthNoiseFactor,2) ;

    // compute birth covariance
    feature_birth.cov[0] = pow(J[0],2)*var_range +
            pow(J[2],2)*var_bearing ;
    feature_birth.cov[1] = J[0]*J[1]*var_range +
            J[2]*J[3]*var_bearing ;
    feature_birth.cov[2] =
            feature_birth.cov[1] ;
    feature_birth.cov[3] = pow(J[1],2)*var_range +
            pow(J[3],2)*var_bearing ;

    // set birth weight
    if(z.label==STATIC_MEASUREMENT || !dev_config.labeledMeasurements)
        feature_birth.weight = safeLog(dev_config.birthWeight) ;
    else
        feature_birth.weight = safeLog(0) ;
}

__device__ void
computeBirth( ConstantVelocityState pose, RangeBearingMeasurement z,
              Gaussian4D& feature_birth)
{
    // invert measurement
    REAL theta = pose.ptheta + z.bearing ;
    REAL dx = z.range*cos(theta) ;
    REAL dy = z.range*sin(theta) ;
    feature_birth.mean[0] = pose.px + dx ;
    feature_birth.mean[1] = pose.py + dy ;

    // inverse measurement jacobian
    REAL J[4] ;
    J[0] = dx/z.range ;
    J[1] = dy/z.range ;
    J[2] = -dy ;
    J[3] = dx ;

    // measurement noise
    REAL var_range = pow(dev_config.stdRange*dev_config.birthNoiseFactor,2) ;
    REAL var_bearing = pow(dev_config.stdBearing*dev_config.birthNoiseFactor,2) ;

    // mean birth velocity is zero
    feature_birth.mean[2] = 0 ;
    feature_birth.mean[3] = 0 ;

    // upper 2x2 block of covariance matrix = K*R*K'
    feature_birth.cov[0] = pow(J[0],2)*var_range +
            pow(J[2],2)*var_bearing ;
    feature_birth.cov[1] = J[0]*J[1]*var_range +
            J[2]*J[3]*var_bearing ;
    feature_birth.cov[4] =
            feature_birth.cov[1] ;
    feature_birth.cov[5] = pow(J[1],2)*var_range +
            pow(J[3],2)*var_bearing ;
    // lower 2 diagonal terms set to parameter value
    feature_birth.cov[10] = dev_config.covVxBirth ;
    feature_birth.cov[15] = dev_config.covVyBirth ;
    // everything else set to 0
    feature_birth.cov[2] = 0 ;
    feature_birth.cov[3] = 0 ;
    feature_birth.cov[6] = 0 ;
    feature_birth.cov[7] = 0 ;
    feature_birth.cov[8] = 0 ;
    feature_birth.cov[9] = 0 ;
    feature_birth.cov[11] = 0 ;
    feature_birth.cov[12] = 0 ;
    feature_birth.cov[13] = 0 ;
    feature_birth.cov[14] = 0 ;

    // set birth weight
    if (z.label == DYNAMIC_MEASUREMENT || !dev_config.labeledMeasurements)
        feature_birth.weight = safeLog(dev_config.birthWeight) ;
    else
        feature_birth.weight = safeLog(0) ;
}

__device__ void
computePreUpdate( ConstantVelocityState pose, Gaussian2D feature_predict,
                  int n_features, int n_measure, REAL& feature_pd,
                  Gaussian2D& feature_nondetect,
                  Gaussian2D*& features_update)
{
    // predicted measurement
    REAL dx = feature_predict.mean[0] - pose.px ;
    REAL dy = feature_predict.mean[1] - pose.py ;
    REAL r2 = dx*dx + dy*dy ;
    REAL r = sqrt(r2) ;
    REAL bearing = wrapAngle(atan2f(dy,dx) - pose.ptheta) ;

    // probability of detection
    feature_pd = 0 ;
    if ( r <= dev_config.maxRange && fabsf(bearing) <= dev_config.maxBearing )
        feature_pd = dev_config.pd ;

    // write non-detection term
    copy_gaussians(feature_predict,feature_nondetect) ;
    feature_nondetect.weight = feature_predict.weight*(1-feature_pd) ;

    // measurement jacobian wrt feature
    REAL J[4] ;
    J[0] = dx/r ;
    J[2] = dy/r ;
    J[1] = -dy/r2 ;
    J[3] = dx/r2 ;

    REAL* P = feature_predict.cov ;

    // BEGIN Maple-Generated expressions
    // innovation covariance
    REAL sigma[4] ;
    sigma[0] = (P[0] * J[0] + J[2] * P[1]) * J[0] + (J[0] * P[2] + P[3] * J[2]) * J[2] + pow(dev_config.stdRange,2) ;
    sigma[1] = (P[0] * J[1] + J[3] * P[1]) * J[0] + (J[1] * P[2] + P[3] * J[3]) * J[2];
    sigma[2] = (P[0] * J[0] + J[2] * P[1]) * J[1] + (J[0] * P[2] + P[3] * J[2]) * J[3];
    sigma[3] = (P[0] * J[1] + J[3] * P[1]) * J[1] + (J[1] * P[2] + P[3] * J[3]) * J[3] + pow(dev_config.stdBearing,2) ;

    // enforce symmetry
    sigma[1] = (sigma[1]+sigma[2])/2 ;
    sigma[2] = sigma[1] ;

    REAL det_sigma = sigma[0]*sigma[3] - sigma[1]*sigma[2] ;

    // inverse of sigma
    REAL S[4] ;
    S[0] = sigma[3]/(det_sigma) ;
    S[1] = -sigma[1]/(det_sigma) ;
    S[2] = -sigma[2]/(det_sigma) ;
    S[3] = sigma[0]/(det_sigma) ;

    // Kalman gain
    REAL K[4] ;
    K[0] = S[0]*(P[0]*J[0] + P[2]*J[2]) + S[1]*(P[0]*J[1] + P[2]*J[3]) ;
    K[1] = S[0]*(P[1]*J[0] + P[3]*J[2]) + S[1]*(P[1]*J[1] + P[3]*J[3]) ;
    K[2] = S[2]*(P[0]*J[0] + P[2]*J[2]) + S[3]*(P[0]*J[1] + P[2]*J[3]) ;
    K[3] = S[2]*(P[1]*J[0] + P[3]*J[2]) + S[3]*(P[1]*J[1] + P[3]*J[3]) ;

    REAL cov_update[4] ;
    cov_update[0] = ((1 - K[0] * J[0] - K[2] * J[1]) * P[0] + (-K[0] * J[2] - K[2] * J[3]) * P[1]) * (1 - K[0] * J[0] - K[2] * J[1]) + ((1 - K[0] * J[0] - K[2] * J[1]) * P[2] + (-K[0] * J[2] - K[2] * J[3]) * P[3]) * (-K[0] * J[2] - K[2] * J[3]) + pow(K[0], 2) * dev_config.stdRange*dev_config.stdRange + pow(K[2], 2) * dev_config.stdBearing*dev_config.stdBearing;
    cov_update[2] = ((1 - K[0] * J[0] - K[2] * J[1]) * P[0] + (-K[0] * J[2] - K[2] * J[3]) * P[1]) * (-K[1] * J[0] - K[3] * J[1]) + ((1 - K[0] * J[0] - K[2] * J[1]) * P[2] + (-K[0] * J[2] - K[2] * J[3]) * P[3]) * (1 - K[1] * J[2] - K[3] * J[3]) + K[0] * dev_config.stdRange*dev_config.stdRange * K[1] + K[2] * dev_config.stdBearing*dev_config.stdBearing * K[3];
    cov_update[1] = ((-K[1] * J[0] - K[3] * J[1]) * P[0] + (1 - K[1] * J[2] - K[3] * J[3]) * P[1]) * (1 - K[0] * J[0] - K[2] * J[1]) + ((-K[1] * J[0] - K[3] * J[1]) * P[2] + (1 - K[1] * J[2] - K[3] * J[3]) * P[3]) * (-K[0] * J[2] - K[2] * J[3]) + K[0] * dev_config.stdRange*dev_config.stdRange * K[1] + K[2] * dev_config.stdBearing*dev_config.stdBearing * K[3];
    cov_update[3] = ((-K[1] * J[0] - K[3] * J[1]) * P[0] + (1 - K[1] * J[2] - K[3] * J[3]) * P[1]) * (-K[1] * J[0] - K[3] * J[1]) + ((-K[1] * J[0] - K[3] * J[1]) * P[2] + (1 - K[1] * J[2] - K[3] * J[3]) * P[3]) * (1 - K[1] * J[2] - K[3] * J[3]) + pow(K[1], 2) * dev_config.stdRange*dev_config.stdRange + pow(K[3], 2) * dev_config.stdBearing*dev_config.stdBearing;

    REAL innov[2] ;
    REAL dist = 0 ;
    for ( int m = 0 ; m < n_measure ; m++ )
    {
        int idx = m*n_features ;
        innov[0] = Z[m].range - r ;
        innov[1] = wrapAngle(Z[m].bearing - bearing) ;
        features_update[idx].mean[0] = feature_predict.mean[0] + K[0]*innov[0] + K[2]*innov[1] ;
        features_update[idx].mean[1] = feature_predict.mean[1] + K[1]*innov[0] + K[3]*innov[1] ;
        for ( int i = 0 ; i < 4 ; i++ )
            features_update[idx].cov[i] = cov_update[i] ;
        // compute single object likelihood
        dist = innov[0]*innov[0]*S[0] +
                innov[0]*innov[1]*(S[1] + S[2]) +
                innov[1]*innov[1]*S[3] ;
        if(Z[m].label==STATIC_MEASUREMENT || !dev_config.labeledMeasurements)
        {
            // partially update weight (log-transformed)
            features_update[idx].weight = safeLog(feature_pd)
                    + safeLog(feature_predict.weight)
                    - 0.5*dist
                    - safeLog(2*M_PI)
                    - 0.5*safeLog(det_sigma) ;
        }
        else
        {
            features_update[idx].weight = safeLog(0) ;
        }
    }
}

__device__ void
computePreUpdate( ConstantVelocityState pose, Gaussian4D feature_predict,
                  int n_features, int n_measure, REAL& feature_pd,
                  Gaussian4D& feature_nondetect,
                  Gaussian4D*& features_update)
{
    // predicted measurement
    REAL dx = feature_predict.mean[0] - pose.px ;
    REAL dy = feature_predict.mean[1] - pose.py ;
    REAL r2 = dx*dx + dy*dy ;
    REAL r = sqrt(r2) ;
    REAL bearing = wrapAngle(atan2f(dy,dx) - pose.ptheta) ;

    // probability of detection
    feature_pd = 0 ;
    if ( r <= dev_config.maxRange && fabsf(bearing) <= dev_config.maxBearing )
        feature_pd = dev_config.pd ;

    // write non-detection term
    copy_gaussians(feature_predict,feature_nondetect) ;
    feature_nondetect.weight = feature_predict.weight*(1-feature_pd) ;

    // measurement jacobian wrt feature
    REAL J[4] ;
    J[0] = dx/r ;
    J[2] = dy/r ;
    J[1] = -dy/r2 ;
    J[3] = dx/r2 ;

    REAL* P = feature_predict.cov ;

    // BEGIN Maple-Generated expressions
    // innovation covariance
    REAL sigma[4] ;
    REAL var_range = pow(dev_config.stdRange,2) ;
    REAL var_bearing = pow(dev_config.stdBearing,2) ;
    sigma[0] = J[0] * (P[0] * J[0] + P[4] * J[2]) + J[2] * (P[1] * J[0] + P[5] * J[2]) + var_range;
    sigma[1] = J[1] * (P[0] * J[0] + P[4] * J[2]) + J[3] * (P[1] * J[0] + P[5] * J[2]);
    sigma[2] = J[0] * (P[0] * J[1] + P[4] * J[3]) + J[2] * (P[1] * J[1] + P[5] * J[3]);
    sigma[3] = J[1] * (P[0] * J[1] + P[4] * J[3]) + J[3] * (P[1] * J[1] + P[5] * J[3]) + var_bearing;

    // enforce symmetry
    sigma[1] = (sigma[1]+sigma[2])/2 ;
    sigma[2] = sigma[1] ;
//			makePositiveDefinite(sigma) ;

    REAL det_sigma = sigma[0]*sigma[3] - sigma[1]*sigma[2] ;

    REAL S[4] ;
    S[0] = sigma[3]/(det_sigma) ;
    S[1] = -sigma[1]/(det_sigma) ;
    S[2] = -sigma[2]/(det_sigma) ;
    S[3] = sigma[0]/(det_sigma) ;

    // Kalman gain
    REAL K[8] ;
    K[0] = P[0] * (J[0] * S[0] + J[1] * S[1])
            + P[4] * (J[2] * S[0] + J[3] * S[1]);
    K[1] = P[1] * (J[0] * S[0] + J[1] * S[1])
            + P[5] * (J[2] * S[0] + J[3] * S[1]);
    K[2] = P[2] * (J[0] * S[0] + J[1] * S[1])
            + P[6] * (J[2] * S[0] + J[3] * S[1]);
    K[3] = P[3] * (J[0] * S[0] + J[1] * S[1])
            + P[7] * (J[2] * S[0] + J[3] * S[1]);
    K[4] = P[0] * (J[0] * S[2] + J[1] * S[3])
            + P[4] * (J[2] * S[2] + J[3] * S[3]);
    K[5] = P[1] * (J[0] * S[2] + J[1] * S[3])
            + P[5] * (J[2] * S[2] + J[3] * S[3]);
    K[6] = P[2] * (J[0] * S[2] + J[1] * S[3])
            + P[6] * (J[2] * S[2] + J[3] * S[3]);
    K[7] = P[3] * (J[0] * S[2] + J[1] * S[3])
            + P[7] * (J[2] * S[2] + J[3] * S[3]);

    // Updated covariance (Joseph Form)
    REAL cov_update[16] ;
    cov_update[0] = (1 - K[0] * J[0] - K[4] * J[1]) * (P[0] * (1 - K[0] * J[0] - K[4] * J[1]) + P[4] * (-K[0] * J[2] - K[4] * J[3])) + (-K[0] * J[2] - K[4] * J[3]) * (P[1] * (1 - K[0] * J[0] - K[4] * J[1]) + P[5] * (-K[0] * J[2] - K[4] * J[3])) + var_range *  pow( K[0],  2) + var_bearing *  pow( K[4],  2);
    cov_update[1] = (-K[1] * J[0] - K[5] * J[1]) * (P[0] * (1 - K[0] * J[0] - K[4] * J[1]) + P[4] * (-K[0] * J[2] - K[4] * J[3])) + (1 - K[1] * J[2] - K[5] * J[3]) * (P[1] * (1 - K[0] * J[0] - K[4] * J[1]) + P[5] * (-K[0] * J[2] - K[4] * J[3])) + K[0] * var_range * K[1] + K[4] * var_bearing * K[5];
    cov_update[2] = (-K[2] * J[0] - K[6] * J[1]) * (P[0] * (1 - K[0] * J[0] - K[4] * J[1]) + P[4] * (-K[0] * J[2] - K[4] * J[3])) + (-K[2] * J[2] - K[6] * J[3]) * (P[1] * (1 - K[0] * J[0] - K[4] * J[1]) + P[5] * (-K[0] * J[2] - K[4] * J[3])) + P[2] * (1 - K[0] * J[0] - K[4] * J[1]) + P[6] * (-K[0] * J[2] - K[4] * J[3]) + K[0] * var_range * K[2] + K[4] * var_bearing * K[6];
    cov_update[3] = (-K[3] * J[0] - K[7] * J[1]) * (P[0] * (1 - K[0] * J[0] - K[4] * J[1]) + P[4] * (-K[0] * J[2] - K[4] * J[3])) + (-K[3] * J[2] - K[7] * J[3]) * (P[1] * (1 - K[0] * J[0] - K[4] * J[1]) + P[5] * (-K[0] * J[2] - K[4] * J[3])) + P[3] * (1 - K[0] * J[0] - K[4] * J[1]) + P[7] * (-K[0] * J[2] - K[4] * J[3]) + K[0] * var_range * K[3] + K[4] * var_bearing * K[7];
    cov_update[4] = (1 - K[0] * J[0] - K[4] * J[1]) * (P[0] * (-K[1] * J[0] - K[5] * J[1]) + P[4] * (1 - K[1] * J[2] - K[5] * J[3])) + (-K[0] * J[2] - K[4] * J[3]) * (P[1] * (-K[1] * J[0] - K[5] * J[1]) + P[5] * (1 - K[1] * J[2] - K[5] * J[3])) + K[0] * var_range * K[1] + K[4] * var_bearing * K[5];
    cov_update[5] = (-K[1] * J[0] - K[5] * J[1]) * (P[0] * (-K[1] * J[0] - K[5] * J[1]) + P[4] * (1 - K[1] * J[2] - K[5] * J[3])) + (1 - K[1] * J[2] - K[5] * J[3]) * (P[1] * (-K[1] * J[0] - K[5] * J[1]) + P[5] * (1 - K[1] * J[2] - K[5] * J[3])) + var_range *  pow( K[1],  2) + var_bearing *  pow( K[5],  2);
    cov_update[6] = (-K[2] * J[0] - K[6] * J[1]) * (P[0] * (-K[1] * J[0] - K[5] * J[1]) + P[4] * (1 - K[1] * J[2] - K[5] * J[3])) + (-K[2] * J[2] - K[6] * J[3]) * (P[1] * (-K[1] * J[0] - K[5] * J[1]) + P[5] * (1 - K[1] * J[2] - K[5] * J[3])) + P[2] * (-K[1] * J[0] - K[5] * J[1]) + P[6] * (1 - K[1] * J[2] - K[5] * J[3]) + K[1] * var_range * K[2] + K[5] * var_bearing * K[6];
    cov_update[7] = (-K[3] * J[0] - K[7] * J[1]) * (P[0] * (-K[1] * J[0] - K[5] * J[1]) + P[4] * (1 - K[1] * J[2] - K[5] * J[3])) + (-K[3] * J[2] - K[7] * J[3]) * (P[1] * (-K[1] * J[0] - K[5] * J[1]) + P[5] * (1 - K[1] * J[2] - K[5] * J[3])) + P[3] * (-K[1] * J[0] - K[5] * J[1]) + P[7] * (1 - K[1] * J[2] - K[5] * J[3]) + K[1] * var_range * K[3] + K[5] * var_bearing * K[7];
    cov_update[8] = (1 - K[0] * J[0] - K[4] * J[1]) * (P[0] * (-K[2] * J[0] - K[6] * J[1]) + P[4] * (-K[2] * J[2] - K[6] * J[3]) + P[8]) + (-K[0] * J[2] - K[4] * J[3]) * (P[1] * (-K[2] * J[0] - K[6] * J[1]) + P[5] * (-K[2] * J[2] - K[6] * J[3]) + P[9]) + K[0] * var_range * K[2] + K[4] * var_bearing * K[6];
    cov_update[9] = (-K[1] * J[0] - K[5] * J[1]) * (P[0] * (-K[2] * J[0] - K[6] * J[1]) + P[4] * (-K[2] * J[2] - K[6] * J[3]) + P[8]) + (1 - K[1] * J[2] - K[5] * J[3]) * (P[1] * (-K[2] * J[0] - K[6] * J[1]) + P[5] * (-K[2] * J[2] - K[6] * J[3]) + P[9]) + K[1] * var_range * K[2] + K[5] * var_bearing * K[6];
    cov_update[10] = (-K[2] * J[0] - K[6] * J[1]) * (P[0] * (-K[2] * J[0] - K[6] * J[1]) + P[4] * (-K[2] * J[2] - K[6] * J[3]) + P[8]) + (-K[2] * J[2] - K[6] * J[3]) * (P[1] * (-K[2] * J[0] - K[6] * J[1]) + P[5] * (-K[2] * J[2] - K[6] * J[3]) + P[9]) + P[2] * (-K[2] * J[0] - K[6] * J[1]) + P[6] * (-K[2] * J[2] - K[6] * J[3]) + P[10] + var_range *  pow( K[2],  2) + var_bearing *  pow( K[6],  2);
    cov_update[11] = (-K[3] * J[0] - K[7] * J[1]) * (P[0] * (-K[2] * J[0] - K[6] * J[1]) + P[4] * (-K[2] * J[2] - K[6] * J[3]) + P[8]) + (-K[3] * J[2] - K[7] * J[3]) * (P[1] * (-K[2] * J[0] - K[6] * J[1]) + P[5] * (-K[2] * J[2] - K[6] * J[3]) + P[9]) + P[3] * (-K[2] * J[0] - K[6] * J[1]) + P[7] * (-K[2] * J[2] - K[6] * J[3]) + P[11] + K[2] * var_range * K[3] + K[6] * var_bearing * K[7];
    cov_update[12] = (1 - K[0] * J[0] - K[4] * J[1]) * (P[0] * (-K[3] * J[0] - K[7] * J[1]) + P[4] * (-K[3] * J[2] - K[7] * J[3]) + P[12]) + (-K[0] * J[2] - K[4] * J[3]) * (P[1] * (-K[3] * J[0] - K[7] * J[1]) + P[5] * (-K[3] * J[2] - K[7] * J[3]) + P[13]) + K[0] * var_range * K[3] + K[4] * var_bearing * K[7];
    cov_update[13] = (-K[1] * J[0] - K[5] * J[1]) * (P[0] * (-K[3] * J[0] - K[7] * J[1]) + P[4] * (-K[3] * J[2] - K[7] * J[3]) + P[12]) + (1 - K[1] * J[2] - K[5] * J[3]) * (P[1] * (-K[3] * J[0] - K[7] * J[1]) + P[5] * (-K[3] * J[2] - K[7] * J[3]) + P[13]) + K[1] * var_range * K[3] + K[5] * var_bearing * K[7];
    cov_update[14] = (-K[2] * J[0] - K[6] * J[1]) * (P[0] * (-K[3] * J[0] - K[7] * J[1]) + P[4] * (-K[3] * J[2] - K[7] * J[3]) + P[12]) + (-K[2] * J[2] - K[6] * J[3]) * (P[1] * (-K[3] * J[0] - K[7] * J[1]) + P[5] * (-K[3] * J[2] - K[7] * J[3]) + P[13]) + P[2] * (-K[3] * J[0] - K[7] * J[1]) + P[6] * (-K[3] * J[2] - K[7] * J[3]) + P[14] + K[2] * var_range * K[3] + K[6] * var_bearing * K[7];
    cov_update[15] = (-K[3] * J[0] - K[7] * J[1]) * (P[0] * (-K[3] * J[0] - K[7] * J[1]) + P[4] * (-K[3] * J[2] - K[7] * J[3]) + P[12]) + (-K[3] * J[2] - K[7] * J[3]) * (P[1] * (-K[3] * J[0] - K[7] * J[1]) + P[5] * (-K[3] * J[2] - K[7] * J[3]) + P[13]) + P[3] * (-K[3] * J[0] - K[7] * J[1]) + P[7] * (-K[3] * J[2] - K[7] * J[3]) + P[15] + var_range *  pow( K[3],  2) + var_bearing *  pow( K[7],  2);

    REAL innov[2] ;
    REAL dist = 0 ;
    for ( int m = 0 ; m < n_measure ; m++ )
    {
        int idx = m*n_features ;

        innov[0] = Z[m].range - r ;
        innov[1] = wrapAngle(Z[m].bearing - bearing) ;
        features_update[idx].mean[0] = feature_predict.mean[0] + K[0]*innov[0] + K[4]*innov[1] ;
        features_update[idx].mean[1] = feature_predict.mean[1] + K[1]*innov[0] + K[5]*innov[1] ;
        features_update[idx].mean[2] = feature_predict.mean[2] + K[2]*innov[0] + K[6]*innov[1] ;
        features_update[idx].mean[3] = feature_predict.mean[3] + K[3]*innov[0] + K[7]*innov[1] ;
        for ( int i = 0 ; i < 16 ; i++ )
            features_update[idx].cov[i] = cov_update[i] ;
        // compute single object likelihood
        dist = innov[0]*innov[0]*S[0] +
                innov[0]*innov[1]*(S[1] + S[2]) +
                innov[1]*innov[1]*S[3] ;
        if(Z[m].label==DYNAMIC_MEASUREMENT || !dev_config.labeledMeasurements)
        {
            // partially update weight (log-transformed)
            features_update[idx].weight = safeLog(feature_pd)
                    + safeLog(feature_predict.weight)
                    - 0.5*dist
                    - safeLog(2*M_PI)
                    - 0.5*safeLog(det_sigma) ;
        }
        else
        {
            features_update[idx].weight = safeLog(0) ;
        }
    }
}


///// computes various components for the Kalman update of a particular feature
///*!
//  * Given a vehicle pose and feature Gaussian, the function computes the Kalman
//  * gain, updated covariance, innovation covariance, determinant of the
//  * innovation covariance, probability of detection, and predicted measurement.
//  * The computed values are stored at the addresses referenced by the passed
//  * pointers.
//  *
//  * This code is specific to XY-heading vehicle state with range-bearing
//  * measurements to XY point features.
//  \param pose vehicle pose
//  \param feature feature gaussian
//  \param K pointer to store Kalman gain matrix
//  \param cov_update pointer to store updated covariance matrix
//  \param det_sigma pointer to store determinant of innov. covariance
//  \param S pointer to store innov. covariance matrix.
//  \param feature_pd pointer to store feature probability of detect.
//  \param z_predict pointer to store predicted measurement
//  */
//__device__ void
//computePreUpdateComponents( ConstantVelocityState pose,
//                            Gaussian2D feature, REAL* K,
//                            REAL* cov_update, REAL* det_sigma,
//                            REAL* S, REAL* feature_pd,
//                            RangeBearingMeasurement* z_predict )
//{
//    // predicted measurement
//    REAL dx = feature.mean[0] - pose.px ;
//    REAL dy = feature.mean[1] - pose.py ;
//    REAL r2 = dx*dx + dy*dy ;
//    REAL r = sqrt(r2) ;
//    REAL bearing = wrapAngle(atan2f(dy,dx) - pose.ptheta) ;

//    z_predict->range = r ;
//    z_predict->bearing = bearing ;

//    // probability of detection
//    if ( r <= dev_config.maxRange && fabsf(bearing) <= dev_config.maxBearing )
//        *feature_pd = dev_config.pd ;
//    else
//        *feature_pd = 0 ;

//    // measurement jacobian wrt feature
//    REAL J[4] ;
//    J[0] = dx/r ;
//    J[2] = dy/r ;
//    J[1] = -dy/r2 ;
//    J[3] = dx/r2 ;

//    // predicted feature covariance
//    REAL* P = feature.cov ;

//    // BEGIN Maple-Generated expressions
//    // innovation covariance
//    REAL sigma[4] ;
//    sigma[0] = (P[0] * J[0] + J[2] * P[1]) * J[0] + (J[0] * P[2] + P[3] * J[2]) * J[2] + pow(dev_config.stdRange,2) ;
//    sigma[1] = (P[0] * J[1] + J[3] * P[1]) * J[0] + (J[1] * P[2] + P[3] * J[3]) * J[2];
//    sigma[2] = (P[0] * J[0] + J[2] * P[1]) * J[1] + (J[0] * P[2] + P[3] * J[2]) * J[3];
//    sigma[3] = (P[0] * J[1] + J[3] * P[1]) * J[1] + (J[1] * P[2] + P[3] * J[3]) * J[3] + pow(dev_config.stdBearing,2) ;

//    // enforce symmetry
//    sigma[1] = (sigma[1]+sigma[2])/2 ;
//    sigma[2] = sigma[1] ;
////			makePositiveDefinite(sigma) ;

//    *det_sigma = sigma[0]*sigma[3] - sigma[1]*sigma[2] ;

//    S[0] = sigma[3]/(*det_sigma) ;
//    S[1] = -sigma[1]/(*det_sigma) ;
//    S[2] = -sigma[2]/(*det_sigma) ;
//    S[3] = sigma[0]/(*det_sigma) ;

//    // Kalman gain
//    K[0] = S[0]*(P[0]*J[0] + P[2]*J[2]) + S[1]*(P[0]*J[1] + P[2]*J[3]) ;
//    K[1] = S[0]*(P[1]*J[0] + P[3]*J[2]) + S[1]*(P[1]*J[1] + P[3]*J[3]) ;
//    K[2] = S[2]*(P[0]*J[0] + P[2]*J[2]) + S[3]*(P[0]*J[1] + P[2]*J[3]) ;
//    K[3] = S[2]*(P[1]*J[0] + P[3]*J[2]) + S[3]*(P[1]*J[1] + P[3]*J[3]) ;

//    // Updated covariance (Joseph Form)
//    cov_update[0] = ((1 - K[0] * J[0] - K[2] * J[1]) * P[0] + (-K[0] * J[2] - K[2] * J[3]) * P[1]) * (1 - K[0] * J[0] - K[2] * J[1]) + ((1 - K[0] * J[0] - K[2] * J[1]) * P[2] + (-K[0] * J[2] - K[2] * J[3]) * P[3]) * (-K[0] * J[2] - K[2] * J[3]) + pow(K[0], 2) * dev_config.stdRange*dev_config.stdRange + pow(K[2], 2) * dev_config.stdBearing*dev_config.stdBearing;
//    cov_update[2] = ((1 - K[0] * J[0] - K[2] * J[1]) * P[0] + (-K[0] * J[2] - K[2] * J[3]) * P[1]) * (-K[1] * J[0] - K[3] * J[1]) + ((1 - K[0] * J[0] - K[2] * J[1]) * P[2] + (-K[0] * J[2] - K[2] * J[3]) * P[3]) * (1 - K[1] * J[2] - K[3] * J[3]) + K[0] * dev_config.stdRange*dev_config.stdRange * K[1] + K[2] * dev_config.stdBearing*dev_config.stdBearing * K[3];
//    cov_update[1] = ((-K[1] * J[0] - K[3] * J[1]) * P[0] + (1 - K[1] * J[2] - K[3] * J[3]) * P[1]) * (1 - K[0] * J[0] - K[2] * J[1]) + ((-K[1] * J[0] - K[3] * J[1]) * P[2] + (1 - K[1] * J[2] - K[3] * J[3]) * P[3]) * (-K[0] * J[2] - K[2] * J[3]) + K[0] * dev_config.stdRange*dev_config.stdRange * K[1] + K[2] * dev_config.stdBearing*dev_config.stdBearing * K[3];
//    cov_update[3] = ((-K[1] * J[0] - K[3] * J[1]) * P[0] + (1 - K[1] * J[2] - K[3] * J[3]) * P[1]) * (-K[1] * J[0] - K[3] * J[1]) + ((-K[1] * J[0] - K[3] * J[1]) * P[2] + (1 - K[1] * J[2] - K[3] * J[3]) * P[3]) * (1 - K[1] * J[2] - K[3] * J[3]) + pow(K[1], 2) * dev_config.stdRange*dev_config.stdRange + pow(K[3], 2) * dev_config.stdBearing*dev_config.stdBearing;
//}

//__device__ void
//computePreUpdateComponentsDynamic( ConstantVelocityState pose,
//                            Gaussian4D feature, REAL* K,
//                            REAL* cov_update, REAL* det_sigma,
//                            REAL* S, REAL* feature_pd,
//                            RangeBearingMeasurement* z_predict )
//{
//    // predicted measurement
//    REAL dx = feature.mean[0] - pose.px ;
//    REAL dy = feature.mean[1] - pose.py ;
//    REAL r2 = dx*dx + dy*dy ;
//    REAL r = sqrt(r2) ;
//    REAL bearing = wrapAngle(atan2f(dy,dx) - pose.ptheta) ;

//    z_predict->range = r ;
//    z_predict->bearing = bearing ;

//    // probability of detection
//    if ( r <= dev_config.maxRange && fabsf(bearing) <= dev_config.maxBearing )
//        *feature_pd = dev_config.pd ;
//    else
//        *feature_pd = 0 ;

//    // measurement jacobian wrt feature
//    REAL J[4] ;
//    J[0] = dx/r ;
//    J[2] = dy/r ;
//    J[1] = -dy/r2 ;
//    J[3] = dx/r2 ;

//    // predicted feature covariance
//    REAL* P = feature.cov ;

//    // BEGIN Maple-Generated expressions
//    // innovation covariance
//    REAL sigma[4] ;
//    REAL var_range = pow(dev_config.stdRange,2) ;
//    REAL var_bearing = pow(dev_config.stdBearing,2) ;
//    sigma[0] = J[0] * (P[0] * J[0] + P[4] * J[2]) + J[2] * (P[1] * J[0] + P[5] * J[2]) + var_range;
//    sigma[1] = J[1] * (P[0] * J[0] + P[4] * J[2]) + J[3] * (P[1] * J[0] + P[5] * J[2]);
//    sigma[2] = J[0] * (P[0] * J[1] + P[4] * J[3]) + J[2] * (P[1] * J[1] + P[5] * J[3]);
//    sigma[3] = J[1] * (P[0] * J[1] + P[4] * J[3]) + J[3] * (P[1] * J[1] + P[5] * J[3]) + var_bearing;

//    // enforce symmetry
//    sigma[1] = (sigma[1]+sigma[2])/2 ;
//    sigma[2] = sigma[1] ;
////			makePositiveDefinite(sigma) ;

//    *det_sigma = sigma[0]*sigma[3] - sigma[1]*sigma[2] ;

//    S[0] = sigma[3]/(*det_sigma) ;
//    S[1] = -sigma[1]/(*det_sigma) ;
//    S[2] = -sigma[2]/(*det_sigma) ;
//    S[3] = sigma[0]/(*det_sigma) ;

//    // Kalman gain
//    K[0] = P[0] * (J[0] * S[0] + J[1] * S[1])
//            + P[4] * (J[2] * S[0] + J[3] * S[1]);
//    K[1] = P[1] * (J[0] * S[0] + J[1] * S[1])
//            + P[5] * (J[2] * S[0] + J[3] * S[1]);
//    K[2] = P[2] * (J[0] * S[0] + J[1] * S[1])
//            + P[6] * (J[2] * S[0] + J[3] * S[1]);
//    K[3] = P[3] * (J[0] * S[0] + J[1] * S[1])
//            + P[7] * (J[2] * S[0] + J[3] * S[1]);
//    K[4] = P[0] * (J[0] * S[2] + J[1] * S[3])
//            + P[4] * (J[2] * S[2] + J[3] * S[3]);
//    K[5] = P[1] * (J[0] * S[2] + J[1] * S[3])
//            + P[5] * (J[2] * S[2] + J[3] * S[3]);
//    K[6] = P[2] * (J[0] * S[2] + J[1] * S[3])
//            + P[6] * (J[2] * S[2] + J[3] * S[3]);
//    K[7] = P[3] * (J[0] * S[2] + J[1] * S[3])
//            + P[7] * (J[2] * S[2] + J[3] * S[3]);

//    // Updated covariance (Joseph Form)
//    cov_update[0] = (1 - K[0] * J[0] - K[4] * J[1]) * (P[0] * (1 - K[0] * J[0] - K[4] * J[1]) + P[4] * (-K[0] * J[2] - K[4] * J[3])) + (-K[0] * J[2] - K[4] * J[3]) * (P[1] * (1 - K[0] * J[0] - K[4] * J[1]) + P[5] * (-K[0] * J[2] - K[4] * J[3])) + var_range *  pow( K[0],  2) + var_bearing *  pow( K[4],  2);
//    cov_update[1] = (-K[1] * J[0] - K[5] * J[1]) * (P[0] * (1 - K[0] * J[0] - K[4] * J[1]) + P[4] * (-K[0] * J[2] - K[4] * J[3])) + (1 - K[1] * J[2] - K[5] * J[3]) * (P[1] * (1 - K[0] * J[0] - K[4] * J[1]) + P[5] * (-K[0] * J[2] - K[4] * J[3])) + K[0] * var_range * K[1] + K[4] * var_bearing * K[5];
//    cov_update[2] = (-K[2] * J[0] - K[6] * J[1]) * (P[0] * (1 - K[0] * J[0] - K[4] * J[1]) + P[4] * (-K[0] * J[2] - K[4] * J[3])) + (-K[2] * J[2] - K[6] * J[3]) * (P[1] * (1 - K[0] * J[0] - K[4] * J[1]) + P[5] * (-K[0] * J[2] - K[4] * J[3])) + P[2] * (1 - K[0] * J[0] - K[4] * J[1]) + P[6] * (-K[0] * J[2] - K[4] * J[3]) + K[0] * var_range * K[2] + K[4] * var_bearing * K[6];
//    cov_update[3] = (-K[3] * J[0] - K[7] * J[1]) * (P[0] * (1 - K[0] * J[0] - K[4] * J[1]) + P[4] * (-K[0] * J[2] - K[4] * J[3])) + (-K[3] * J[2] - K[7] * J[3]) * (P[1] * (1 - K[0] * J[0] - K[4] * J[1]) + P[5] * (-K[0] * J[2] - K[4] * J[3])) + P[3] * (1 - K[0] * J[0] - K[4] * J[1]) + P[7] * (-K[0] * J[2] - K[4] * J[3]) + K[0] * var_range * K[3] + K[4] * var_bearing * K[7];
//    cov_update[4] = (1 - K[0] * J[0] - K[4] * J[1]) * (P[0] * (-K[1] * J[0] - K[5] * J[1]) + P[4] * (1 - K[1] * J[2] - K[5] * J[3])) + (-K[0] * J[2] - K[4] * J[3]) * (P[1] * (-K[1] * J[0] - K[5] * J[1]) + P[5] * (1 - K[1] * J[2] - K[5] * J[3])) + K[0] * var_range * K[1] + K[4] * var_bearing * K[5];
//    cov_update[5] = (-K[1] * J[0] - K[5] * J[1]) * (P[0] * (-K[1] * J[0] - K[5] * J[1]) + P[4] * (1 - K[1] * J[2] - K[5] * J[3])) + (1 - K[1] * J[2] - K[5] * J[3]) * (P[1] * (-K[1] * J[0] - K[5] * J[1]) + P[5] * (1 - K[1] * J[2] - K[5] * J[3])) + var_range *  pow( K[1],  2) + var_bearing *  pow( K[5],  2);
//    cov_update[6] = (-K[2] * J[0] - K[6] * J[1]) * (P[0] * (-K[1] * J[0] - K[5] * J[1]) + P[4] * (1 - K[1] * J[2] - K[5] * J[3])) + (-K[2] * J[2] - K[6] * J[3]) * (P[1] * (-K[1] * J[0] - K[5] * J[1]) + P[5] * (1 - K[1] * J[2] - K[5] * J[3])) + P[2] * (-K[1] * J[0] - K[5] * J[1]) + P[6] * (1 - K[1] * J[2] - K[5] * J[3]) + K[1] * var_range * K[2] + K[5] * var_bearing * K[6];
//    cov_update[7] = (-K[3] * J[0] - K[7] * J[1]) * (P[0] * (-K[1] * J[0] - K[5] * J[1]) + P[4] * (1 - K[1] * J[2] - K[5] * J[3])) + (-K[3] * J[2] - K[7] * J[3]) * (P[1] * (-K[1] * J[0] - K[5] * J[1]) + P[5] * (1 - K[1] * J[2] - K[5] * J[3])) + P[3] * (-K[1] * J[0] - K[5] * J[1]) + P[7] * (1 - K[1] * J[2] - K[5] * J[3]) + K[1] * var_range * K[3] + K[5] * var_bearing * K[7];
//    cov_update[8] = (1 - K[0] * J[0] - K[4] * J[1]) * (P[0] * (-K[2] * J[0] - K[6] * J[1]) + P[4] * (-K[2] * J[2] - K[6] * J[3]) + P[8]) + (-K[0] * J[2] - K[4] * J[3]) * (P[1] * (-K[2] * J[0] - K[6] * J[1]) + P[5] * (-K[2] * J[2] - K[6] * J[3]) + P[9]) + K[0] * var_range * K[2] + K[4] * var_bearing * K[6];
//    cov_update[9] = (-K[1] * J[0] - K[5] * J[1]) * (P[0] * (-K[2] * J[0] - K[6] * J[1]) + P[4] * (-K[2] * J[2] - K[6] * J[3]) + P[8]) + (1 - K[1] * J[2] - K[5] * J[3]) * (P[1] * (-K[2] * J[0] - K[6] * J[1]) + P[5] * (-K[2] * J[2] - K[6] * J[3]) + P[9]) + K[1] * var_range * K[2] + K[5] * var_bearing * K[6];
//    cov_update[10] = (-K[2] * J[0] - K[6] * J[1]) * (P[0] * (-K[2] * J[0] - K[6] * J[1]) + P[4] * (-K[2] * J[2] - K[6] * J[3]) + P[8]) + (-K[2] * J[2] - K[6] * J[3]) * (P[1] * (-K[2] * J[0] - K[6] * J[1]) + P[5] * (-K[2] * J[2] - K[6] * J[3]) + P[9]) + P[2] * (-K[2] * J[0] - K[6] * J[1]) + P[6] * (-K[2] * J[2] - K[6] * J[3]) + P[10] + var_range *  pow( K[2],  2) + var_bearing *  pow( K[6],  2);
//    cov_update[11] = (-K[3] * J[0] - K[7] * J[1]) * (P[0] * (-K[2] * J[0] - K[6] * J[1]) + P[4] * (-K[2] * J[2] - K[6] * J[3]) + P[8]) + (-K[3] * J[2] - K[7] * J[3]) * (P[1] * (-K[2] * J[0] - K[6] * J[1]) + P[5] * (-K[2] * J[2] - K[6] * J[3]) + P[9]) + P[3] * (-K[2] * J[0] - K[6] * J[1]) + P[7] * (-K[2] * J[2] - K[6] * J[3]) + P[11] + K[2] * var_range * K[3] + K[6] * var_bearing * K[7];
//    cov_update[12] = (1 - K[0] * J[0] - K[4] * J[1]) * (P[0] * (-K[3] * J[0] - K[7] * J[1]) + P[4] * (-K[3] * J[2] - K[7] * J[3]) + P[12]) + (-K[0] * J[2] - K[4] * J[3]) * (P[1] * (-K[3] * J[0] - K[7] * J[1]) + P[5] * (-K[3] * J[2] - K[7] * J[3]) + P[13]) + K[0] * var_range * K[3] + K[4] * var_bearing * K[7];
//    cov_update[13] = (-K[1] * J[0] - K[5] * J[1]) * (P[0] * (-K[3] * J[0] - K[7] * J[1]) + P[4] * (-K[3] * J[2] - K[7] * J[3]) + P[12]) + (1 - K[1] * J[2] - K[5] * J[3]) * (P[1] * (-K[3] * J[0] - K[7] * J[1]) + P[5] * (-K[3] * J[2] - K[7] * J[3]) + P[13]) + K[1] * var_range * K[3] + K[5] * var_bearing * K[7];
//    cov_update[14] = (-K[2] * J[0] - K[6] * J[1]) * (P[0] * (-K[3] * J[0] - K[7] * J[1]) + P[4] * (-K[3] * J[2] - K[7] * J[3]) + P[12]) + (-K[2] * J[2] - K[6] * J[3]) * (P[1] * (-K[3] * J[0] - K[7] * J[1]) + P[5] * (-K[3] * J[2] - K[7] * J[3]) + P[13]) + P[2] * (-K[3] * J[0] - K[7] * J[1]) + P[6] * (-K[3] * J[2] - K[7] * J[3]) + P[14] + K[2] * var_range * K[3] + K[6] * var_bearing * K[7];
//    cov_update[15] = (-K[3] * J[0] - K[7] * J[1]) * (P[0] * (-K[3] * J[0] - K[7] * J[1]) + P[4] * (-K[3] * J[2] - K[7] * J[3]) + P[12]) + (-K[3] * J[2] - K[7] * J[3]) * (P[1] * (-K[3] * J[0] - K[7] * J[1]) + P[5] * (-K[3] * J[2] - K[7] * J[3]) + P[13]) + P[3] * (-K[3] * J[0] - K[7] * J[1]) + P[7] * (-K[3] * J[2] - K[7] * J[3]) + P[15] + var_range *  pow( K[3],  2) + var_bearing *  pow( K[7],  2);
//}

///// kernel for computing various constants used in the CPHD filter
//__global__ void
//cphdConstantsKernel( REAL* dev_factorial, REAL* dev_C, REAL* dev_cn_clutter )
//{
//    int n = threadIdx.x ;
//    int k = blockIdx.x ;
//    REAL* factorial = (REAL*)shmem ;

//    factorial[n] = dev_factorial[n] ;
//    __syncthreads() ;

//    // compute the log binomial coefficients (nchoosek)
//    int stride = dev_config.maxCardinality + 1 ;
//    int idx = k*stride + n ;
//    REAL log_nchoosek = 0 ;
//    if ( k == 0 )
//    {
//        log_nchoosek = 0 ;
//    }
//    else if ( n == 0 || k > n )
//    {
//        log_nchoosek = LOG0 ;
//    }
//    else
//    {
//        log_nchoosek = factorial[n] - factorial[k]
//                - factorial[n-k] ;
//    }
//    dev_C[idx] = log_nchoosek ;


//    // thread block 0 computes the clutter cardinality
//    if ( k == 0 )
//    {
//        dev_cn_clutter[n] = n*safeLog(dev_config.clutterRate)
//                - dev_config.clutterRate
//                - factorial[n] ;
//    }
////	// for debugging: clutter cardinality with constant number of clutter
////	if ( k== 0 )
////	{
////		if ( n == dev_config.clutterRate)
////			dev_cn_clutter[n] = 0 ;
////		else
////			dev_cn_clutter[n] = LOG0 ;
////	}

//}

///// host-side helper function to call cphdConstantsKernel
//void
//initCphdConstants()
//{
//    log_factorials = (REAL*)malloc( (config.maxCardinality+1)*sizeof(REAL) ) ;
//    log_factorials[0] = 0 ;
//    for ( int n = 1 ; n <= config.maxCardinality ; n++ )
//    {
//        log_factorials[n] = log_factorials[n-1] + safeLog((REAL)n) ;
//    }
//    checkCudaErrors( cudaMalloc( (void**)&dev_C,
//                                pow(config.maxCardinality+1,2)*sizeof(REAL) ) ) ;
//    checkCudaErrors( cudaMalloc( (void**)&dev_factorial,
//                                (config.maxCardinality+1)*sizeof(REAL) ) ) ;
//    checkCudaErrors( cudaMalloc( (void**)&dev_cn_clutter,
//                                (config.maxCardinality+1)*sizeof(REAL) ) ) ;
//    checkCudaErrors( cudaMemcpy( dev_factorial, &log_factorials[0],
//                                (config.maxCardinality+1)*sizeof(REAL),
//                                cudaMemcpyHostToDevice ) ) ;
//    //
////	checkCudaErrors(
////				cudaMalloc( (void**)&dev_pspower,
////							(config.maxCardinality+1)*sizeof(REAL) ) ) ;
////	checkCudaErrors(
////				cudaMalloc( (void**)&dev_qspower,
////							(config.maxCardinality+1)*sizeof(REAL) ) ) ;

//    int n_blocks = config.maxCardinality+1 ;
//    int n_threads = n_blocks ;
//    cphdConstantsKernel<<<n_blocks, n_threads, n_threads*sizeof(REAL)>>>
//        ( dev_factorial, dev_C, dev_cn_clutter ) ;
//    //
//}

/// kernel for particle prediction with an ackerman steering motion model
__global__ void
phdPredictKernelAckerman(ConstantVelocityState* particles_prior,
                   AckermanControl control,
                   AckermanNoise* noise,
                   ConstantVelocityState* particles_predict,
                    int n_predict)
{
    const int tid = threadIdx.x ;
    const int predict_idx = blockIdx.x*blockDim.x + tid ;
    if (predict_idx < n_predict)
    {
        // get the prior state from which this prediction is generated
        const int prior_idx = floor((float)predict_idx/dev_config.nPredictParticles) ;
        ConstantVelocityState oldState = particles_prior[prior_idx] ;

        // use the motion model to compute the prediction
        ConstantVelocityState newState ;
        REAL ve_noisy = control.v_encoder + noise[predict_idx].n_encoder ;
        REAL alpha_noisy = control.alpha + noise[predict_idx].n_alpha ;
        REAL vc = ve_noisy/(1-tan(alpha_noisy)*dev_config.h/dev_config.l) ;
        REAL xc_dot = vc*cos(oldState.ptheta) ;
        REAL yc_dot = vc*sin(oldState.ptheta) ;
        REAL thetac_dot = vc*tan(alpha_noisy)/dev_config.l ;
        REAL dt = dev_config.dt/dev_config.subdividePredict ;
        newState.px = oldState.px +
                dt*(xc_dot -
                thetac_dot*( dev_config.a*sin(oldState.ptheta) + dev_config.b*cos(oldState.ptheta) )
        ) ;
        newState.py = oldState.py +
                dt*(yc_dot +
                thetac_dot*( dev_config.a*cos(oldState.ptheta) - dev_config.b*sin(oldState.ptheta) )
        ) ;
        newState.ptheta = wrapAngle(oldState.ptheta + dt*thetac_dot) ;
        newState.vx = 0 ;
        newState.vy = 0 ;
        newState.vtheta = 0 ;

        // save predicted state to memory
        particles_predict[predict_idx] = newState ;
    }
}

__global__ void
phdPredictKernel(ConstantVelocityState* particles_prior,
        ConstantVelocityNoise* noise, ConstantVelocityState* particles_predict,
        int n_predict)
{
    const int tid = threadIdx.x ;
    const int predict_idx = blockIdx.x*blockDim.x + tid ;
    if (predict_idx < n_predict)
    {
        const int prior_idx = floor((float)predict_idx/dev_config.nPredictParticles) ;
        ConstantVelocityState oldState = particles_prior[prior_idx] ;
        ConstantVelocityState newState ;
        REAL dt = dev_config.dt/dev_config.subdividePredict ;
    //	typename modelType::stateType newState = mm(particles[particleIdx],*control,noise[particleIdx]) ;
        newState.px = oldState.px +
                dt*(oldState.vx*cos(oldState.ptheta) -
                               oldState.vy*sin(oldState.ptheta))+
                dt*dt*0.5*(noise[predict_idx].ax*cos(oldState.ptheta) -
                                                 noise[predict_idx].ay*sin(oldState.ptheta)) ;
        newState.py = oldState.py +
                dt*(oldState.vx*sin(oldState.ptheta) +
                               oldState.vy*cos(oldState.ptheta)) +
                dt*dt*0.5*(noise[predict_idx].ax*sin(oldState.ptheta) +
                                                 noise[predict_idx].ay*cos(oldState.ptheta)) ;
        newState.ptheta = wrapAngle(oldState.ptheta +
                                    dt*oldState.vtheta +
                                    0.5*dt*dt*noise[predict_idx].atheta) ;
        newState.vx = oldState.vx + dt*noise[predict_idx].ax ;
        newState.vy = oldState.vy + dt*noise[predict_idx].ay ;
        newState.vtheta = oldState.vtheta + dt*noise[predict_idx].atheta ;
        particles_predict[predict_idx] = newState ;
    }
}

/// predict the cardinality distribution for the CPHD filter
/**
  Each thread block processes the cardinality for a single particle. Each thread
  inside the block computes the predicted cardinality for a particular value of
  n.
  */
__global__ void
cardinalityPredictKernel( REAL* cn_prior, REAL* cn_births, REAL* dev_C,
                          REAL* cn_predict )
{
    int n = threadIdx.x ;
    int cn_offset = blockIdx.x * (dev_config.maxCardinality+1) ;
    REAL* cn_prior_shared = (REAL*)shmem ;

    // load the prior cardinality into shared mem
    cn_prior_shared[n] = cn_prior[cn_offset+n] ;
    __syncthreads() ;

    REAL outersum = 0 ;
    for ( int j = 0 ; j <= n ; j++ )
    {
        outersum += exp(cn_births[n-j]+cn_prior_shared[j]) ;
    }
    if ( outersum != 0)
        cn_predict[cn_offset+n] = safeLog(outersum) ;
    else
        cn_predict[cn_offset+n] = LOG0 ;
}

/// compute the predicted states of every feature
template<class GaussianType, class MotionModelType>
__global__ void
predictMapKernel(GaussianType* features_prior, MotionModelType model,
                 int n_features, GaussianType* features_predict)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x ;
    // loop over all features
    for (int j = 0 ; j < n_features ; j+=blockDim.x*gridDim.x)
    {
        int idx = j+tid ;
        if ( idx < n_features )
        {
            features_predict[idx] = model.compute_prediction(features_prior[idx],
                                                             dev_config.dt) ;
        }
    }
}


__global__ void
predictMapKernelMixed(Gaussian4D* features_prior,
                      ConstantVelocityMotionModel model,
                      int n_features, Gaussian4D* features_predict,
                      Gaussian2D* features_jump)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x ;
    // loop over all features
    for (int j = 0 ; j < n_features ; j+=blockDim.x*gridDim.x)
    {
        int idx = j+tid ;
        if ( idx < n_features )
        {
            REAL vx = features_prior[idx].mean[2] ;
            REAL vy = features_prior[idx].mean[3] ;
            REAL v_mag = sqrt(vx*vx + vy*vy) ;
            REAL sigmoid_v = 1/(1+exp(dev_config.beta*(dev_config.tau - v_mag))) ;
            REAL p_jmm ;
            REAL ps ;
            REAL scale_x = 1 ;
            REAL scale_y = 1 ;
            if (dev_config.featureModel==DYNAMIC_MODEL)
            {
                p_jmm = 1 ;
                ps = logistic_function(v_mag,0,1-dev_config.ps,dev_config.beta,
                                       dev_config.tau) ;
                ps = 1-ps ;
                scale_x = logistic_function(vx,0,1,dev_config.beta,
                                            dev_config.tau) ;
                scale_y = logistic_function(vy,0,1,dev_config.beta,
                                            dev_config.tau) ;
            }
            else if(dev_config.featureModel==MIXED_MODEL)
            {
                p_jmm = sigmoid_v ;
                ps = dev_config.ps ;
//                p_jmm = 1 ;
            }
            features_predict[idx] = model.compute_prediction(features_prior[idx],
                                                             dev_config.dt,
                                                             scale_x,scale_y) ;
            features_predict[idx].weight = p_jmm*ps
                    *features_predict[idx].weight ;

            features_jump[idx].weight = (1-p_jmm)*features_prior[idx].weight ;
            features_jump[idx].mean[0] = features_prior[idx].mean[0] ;
            features_jump[idx].mean[1] = features_prior[idx].mean[1] ;
            features_jump[idx].cov[0] = features_prior[idx].cov[0] ;
            features_jump[idx].cov[1] = features_prior[idx].cov[1] ;
            features_jump[idx].cov[2] = features_prior[idx].cov[4] ;
            features_jump[idx].cov[3] = features_prior[idx].cov[5] ;
        }
    }
}

void
predictMapMixed(SynthSLAM& particles)
{
    // combine all dynamic features into one vector
    vector<Gaussian4D> all_features = combineFeatures(particles.maps_dynamic) ;
    int n_features = all_features.size() ;

    // allocate memory
    Gaussian4D* dev_features_prior = NULL ;
    Gaussian4D* dev_features_predict = NULL ;
    Gaussian2D* dev_features_jump = NULL ;
    checkCudaErrors(cudaMalloc((void**)&dev_features_prior,
                              n_features*sizeof(Gaussian4D) ) ) ;
    checkCudaErrors(cudaMalloc((void**)&dev_features_predict,
                              n_features*sizeof(Gaussian4D) ) ) ;
    checkCudaErrors(cudaMalloc((void**)&dev_features_jump,
                              n_features*sizeof(Gaussian2D) ) ) ;
    checkCudaErrors(cudaMemcpy(dev_features_prior,&all_features[0],
                              n_features*sizeof(Gaussian4D),
                              cudaMemcpyHostToDevice) ) ;
    int n_blocks = (n_features+255)/256 ;
    // configure the feature motion model
    ConstantVelocityMotionModel motion_model ;
    motion_model.std_accx = config.stdAxMap ;
    motion_model.std_accy = config.stdAyMap ;

    // launch the kernel
    predictMapKernelMixed<<<n_blocks,256>>>
        (dev_features_prior,motion_model,n_features, dev_features_predict,
         dev_features_jump ) ;

    // copy results from device
    vector<Gaussian2D> all_features_jump( all_features.size() ) ;
    checkCudaErrors(cudaMemcpy(&all_features[0],dev_features_predict,
                              n_features*sizeof(Gaussian4D),
                              cudaMemcpyDeviceToHost)) ;
    checkCudaErrors(cudaMemcpy(&all_features_jump[0],dev_features_jump,
                              n_features*sizeof(Gaussian2D),
                              cudaMemcpyDeviceToHost)) ;
    // load predicted features back into particles
    Gaussian4D* begin = &all_features[0] ;
    Gaussian4D* end = begin
            + particles.maps_dynamic[0].size() ;
    Gaussian2D* begin_jump = &all_features_jump[0] ;
    Gaussian2D* end_jump = begin_jump
            + particles.maps_dynamic[0].size() ;

    for ( int n = 0 ; n < particles.n_particles ; n++ )
    {
        particles.maps_dynamic[n].assign(begin,end) ;
//        if(config.featureModel==MIXED_MODEL)
//        {
//            particles.maps_static[n].insert(particles.maps_static[n].end(),
//                                            begin_jump,
//                                            end_jump ) ;
//        }
        if ( n < particles.n_particles - 1)
        {
            begin = end ;
            end += particles.maps_dynamic[n+1].size() ;

            begin_jump = end_jump ;
            end_jump += particles.maps_dynamic[n+1].size() ;
        }
    }

    // free memory
    checkCudaErrors( cudaFree( dev_features_prior ) ) ;
    checkCudaErrors( cudaFree( dev_features_predict ) ) ;
    checkCudaErrors( cudaFree( dev_features_jump ) ) ;
}

//template <class GaussianType>
//void
//predictMap(SynthSLAM& particles)
//{
//    // combine all dynamic features into one vector
//    vector<Gaussian4D> all_features = combineFeatures(particles.maps_dynamic) ;
//    int n_features = all_features.size() ;
//    GaussianType* dev_features_prior = NULL ;
//    GaussianType* dev_features_predict = NULL ;
//    checkCudaErrors(cudaMalloc((void**)&dev_features_prior,
//                              n_features*sizeof(Gaussian4D) ) ) ;
//    checkCudaErrors(cudaMalloc((void**)&dev_features_predict,
//                              n_features*sizeof(Gaussian4D) ) ) ;
//    checkCudaErrors(cudaMemcpy(dev_features_prior,&all_features[0],
//                              n_features*sizeof(Gaussian4D),
//                              cudaMemcpyHostToDevice) ) ;
//    int n_blocks = (n_features+255)/256 ;
//    ConstantVelocityMotionModel motion_model ;
//    motion_model.std_accx = config.stdAxMap ;
//    motion_model.std_accy = config.stdAyMap ;
//    predictMapKernel<<<n_blocks,256>>>
//        (dev_features_prior,motion_model,n_features, dev_features_predict ) ;
//    checkCudaErrors(cudaMemcpy(&all_features[0],dev_features_predict,
//                              n_features*sizeof(GaussianType),
//                              cudaMemcpyDeviceToHost)) ;
//    // load predicted features back into particles
//    GaussianType* begin = &all_features[0] ;
//    GaussianType* end = begin
//            + particles.maps[0].size() ;
//    for ( int n = 0 ; n < particles.n_particles ; n++ )
//    {
//        particles.maps_dynamic[n].assign(begin,end) ;
//        if ( n < particles.n_particles - 1)
//        {
//            begin = end ;
//            end += particles.maps_dynamic[n+1].size() ;
//        }
//    }
//    checkCudaErrors( cudaFree( dev_features_prior ) ) ;
//    checkCudaErrors( cudaFree( dev_features_predict ) ) ;
//}

/// host-side helper function for PHD filter prediction
void
phdPredict(SynthSLAM& particles, ... )
{
    // start timer
    cudaEvent_t start, stop ;
    cudaEventCreate( &start ) ;
    cudaEventCreate( &stop ) ;
    cudaEventRecord( start,0 ) ;

    int n_particles = particles.n_particles ;
    int nPredict = n_particles*config.nPredictParticles ;

    // allocate device memory
    ConstantVelocityState* dev_states_prior = NULL ;
    ConstantVelocityState* dev_states_predict = NULL ;
    checkCudaErrors(
                cudaMalloc((void**)&dev_states_prior,
                           n_particles*sizeof(ConstantVelocityState) ) ) ;
    checkCudaErrors(
                cudaMalloc((void**)&dev_states_predict,
                           nPredict*sizeof(ConstantVelocityState) ) ) ;

    // copy inputs
    checkCudaErrors(
                cudaMemcpy(dev_states_prior, &particles.states[0],
                           n_particles*sizeof(ConstantVelocityState),
                            cudaMemcpyHostToDevice) ) ;


    if ( config.motionType == CV_MOTION )
    {
        // generate random noise values
        std::vector<ConstantVelocityNoise> noiseVector(nPredict) ;
        for (unsigned int i = 0 ; i < nPredict ; i++ )
        {
            noiseVector[i].ax = 3*config.ax * randn() ;
            noiseVector[i].ay = 3*config.ay * randn() ;
            noiseVector[i].atheta = 3*config.ayaw * randn() ;
        }

        ConstantVelocityNoise* dev_noise = NULL ;
        checkCudaErrors(
                    cudaMalloc((void**)&dev_noise,
                               n_particles*sizeof(ConstantVelocityNoise) ) ) ;
        checkCudaErrors(
                    cudaMemcpy(dev_noise, &noiseVector[0],
                               n_particles*sizeof(ConstantVelocityNoise),
                               cudaMemcpyHostToDevice) ) ;

        // launch the kernel
        int nThreads = min(nPredict,256) ;
        int nBlocks = (nPredict+255)/256 ;
        phdPredictKernel
        <<<nBlocks, nThreads>>>
        ( dev_states_prior,dev_noise,dev_states_predict,nPredict ) ;

        cudaFree(dev_noise) ;
    }
    else if( config.motionType == ACKERMAN_MOTION )
    {
        // read in the control data structure from variable argument list
        va_list argptr ;
        va_start(argptr,particles) ;
        AckermanControl control = (AckermanControl)va_arg(argptr,AckermanControl) ;
        va_end(argptr) ;

        // generate random noise values
        std::vector<AckermanNoise> noiseVector(nPredict) ;
        for (unsigned int i = 0 ; i < nPredict ; i++ )
        {
            noiseVector[i].n_alpha = config.stdAlpha * randn() ;
            noiseVector[i].n_encoder = config.stdEncoder * randn() ;
        }
        AckermanNoise* dev_noise = NULL ;
        checkCudaErrors(
                    cudaMalloc((void**)&dev_noise,
                               nPredict*sizeof(AckermanNoise) ) ) ;
        checkCudaErrors(
                    cudaMemcpy(dev_noise, &noiseVector[0],
                               nPredict*sizeof(AckermanNoise),
                               cudaMemcpyHostToDevice) ) ;

        // launch the kernel
        int nThreads = min(nPredict,256) ;
        int nBlocks = (nPredict+255)/256 ;
        phdPredictKernelAckerman
        <<<nBlocks, nThreads>>>
        (dev_states_prior,control,dev_noise,dev_states_predict,nPredict) ;

        cudaFree(dev_noise) ;
    }


    // copy results from device
    ConstantVelocityState* states_predict = (ConstantVelocityState*)
                                            malloc(nPredict*sizeof(ConstantVelocityState)) ;
    checkCudaErrors(cudaMemcpy(states_predict, dev_states_predict,
                              nPredict*sizeof(ConstantVelocityState),
                              cudaMemcpyDeviceToHost) ) ;
    particles.states.assign( states_predict, states_predict+nPredict ) ;



    // duplicate the PHD filter maps and cardinalities for the newly spawned
    // vehicle particles, and downscale particle weights
    if ( config.nPredictParticles > 1 )
    {
        DEBUG_MSG("Duplicating maps") ;
        vector<vector<Gaussian2D> > maps_predict_static ;
        vector<vector<Gaussian4D> > maps_predict_dynamic ;
        vector<REAL> weights_predict ;
        vector< vector <REAL> > cardinalities_predict ;
        vector<int> resample_idx_predict ;
//        maps_predict_static.clear();
//        maps_predict_static.reserve(nPredict);
//        maps_predict_dynamic.clear();
//        maps_predict_dynamic.reserve(nPredict);
//        weights_predict.clear();
//        weights_predict.reserve(nPredict);
//        cardinalities_predict.clear();
//        cardinalities_predict.reserve(nPredict);
//        resample_idx_predict.reserve(nPredict);
        for ( int i = 0 ; i < n_particles ; i++ )
        {

            maps_predict_static.insert( maps_predict_static.end(),
                                        config.nPredictParticles,
                                        particles.maps_static[i] ) ;

            maps_predict_dynamic.insert( maps_predict_dynamic.end(),
                                        config.nPredictParticles,
                                        particles.maps_dynamic[i] ) ;

            cardinalities_predict.insert( cardinalities_predict.end(),
                                          config.nPredictParticles,
                                          particles.cardinalities[i] ) ;

            float new_weight = particles.weights[i] - safeLog(config.nPredictParticles) ;
//            DEBUG_VAL(new_weight) ;
            weights_predict.insert( weights_predict.end(), config.nPredictParticles,
                                    new_weight ) ;

            resample_idx_predict.insert(resample_idx_predict.end(),
                                        config.nPredictParticles,
                                        particles.resample_idx[i]) ;
        }
//        DEBUG_VAL(maps_predict.size()) ;
        DEBUG_MSG("saving duplicated maps") ;
        DEBUG_MSG("static") ;
        particles.maps_static = maps_predict_static ;
        DEBUG_MSG("dynamic") ;
        particles.maps_dynamic = maps_predict_dynamic ;
         DEBUG_MSG("weights") ;
        particles.weights = weights_predict ;
        DEBUG_MSG("cardinalities") ;
        particles.cardinalities = cardinalities_predict ;
        particles.resample_idx = resample_idx_predict ;
        particles.n_particles = nPredict ;
    }

    // map prediction
    if(config.featureModel==DYNAMIC_MODEL || config.featureModel==MIXED_MODEL)
        predictMapMixed(particles) ;

    // log time
    cudaEventRecord( stop,0 ) ;
    cudaEventSynchronize( stop ) ;
    float elapsed ;
    cudaEventElapsedTime( &elapsed, start, stop ) ;
    fstream predictTimeFile( "predicttime.log", fstream::out|fstream::app ) ;
    predictTimeFile << elapsed << endl ;
    predictTimeFile.close() ;

    // clean up
    checkCudaErrors( cudaFree( dev_states_prior ) ) ;
    checkCudaErrors( cudaFree( dev_states_predict ) ) ;
    free(states_predict) ;
}

/// determine which features are in range
/*!
  * Each thread block handles a single particle. The threads in the block
  * evaluate the range and bearing [blockDim] features in parallel, looping
  * through all of the particle's features.

    \param predictedFeatures Features from all particles concatenated into a
        single array
    \param map_sizes_static Number of features in each particle, so that the function
        knows where the boundaries are in predictedFeatures
    \param n_particles Total number of particles
    \param poses Array of particle poses
    \param inRange Pointer to boolean array that is filled by the function.
        For each feature in predictedFeatures that is in range of its
        respective particle, the corresponding entry in this array is set to
        true
    \param nInRange Pointer to integer array that is filled by the function.
        Should be allocated to have [n_particles] elements. Each entry
        represents the number of in range features for each particle.
  */
template<class GaussianType>
__global__ void
computeInRangeKernel( GaussianType *predictedFeatures, int* map_sizes_static,
                      int n_particles, ConstantVelocityState* poses,
                      char* inRange, int* n_in_range, int* n_nearly_in_range )
{
    int tid = threadIdx.x ;

    // total number of predicted features per block
    int n_featuresBlock ;
    // number of inrange features in the particle
    __shared__ int nInRangeBlock ;
    __shared__ int n_nearly_in_range_block ;
    // vehicle pose of the thread block
    ConstantVelocityState blockPose ;

    GaussianType feature ;
    for ( int p = 0 ; p < n_particles ; p += gridDim.x )
    {
        if ( p + blockIdx.x < n_particles )
        {
            int predict_offset = 0 ;
            // compute the indexing offset for this particle
            int map_idx = p + blockIdx.x ;
            for ( int i = 0 ; i < map_idx ; i++ )
                predict_offset += map_sizes_static[i] ;
            // particle-wide values
            if ( tid == 0 )
            {
                nInRangeBlock = 0 ;
                n_nearly_in_range_block = 0 ;
            }
            blockPose = poses[map_idx] ;
            n_featuresBlock = map_sizes_static[map_idx] ;
            __syncthreads() ;

            // loop through features
            for ( int i = 0 ; i < n_featuresBlock ; i += blockDim.x )
            {
                if ( tid+i < n_featuresBlock )
                {
                    // index of thread feature
                    int featureIdx = predict_offset + tid + i ;
                    feature = predictedFeatures[featureIdx] ;

                    // default value
                    inRange[featureIdx] = 0 ;

                    // compute the predicted measurement
                    REAL dx = feature.mean[0] - blockPose.px ;
                    REAL dy = feature.mean[1] - blockPose.py ;
                    REAL r2 = dx*dx + dy*dy ;
                    REAL r = sqrt(r2) ;
                    REAL bearing = wrapAngle(atan2f(dy,dx) - blockPose.ptheta) ;
                    if ( r >= dev_config.minRange &&
                         r <= dev_config.maxRange &&
                         fabs(bearing) <= dev_config.maxBearing )
                    {
                        atomicAdd( &nInRangeBlock, 1 ) ;
                        inRange[featureIdx] = 1 ;
                    }
                    else if ( r >= 0.8*dev_config.minRange &&
                              r <= 1.2*dev_config.maxRange &&
                              fabs(bearing) <= 1.2*dev_config.maxBearing )
                    {
                        inRange[featureIdx] = 2 ;
                        atomicAdd( &n_nearly_in_range_block, 1 ) ;
                    }
                }
            }
            // store nInrange
            __syncthreads() ;
            if ( tid == 0 )
            {
                n_in_range[map_idx] = nInRangeBlock ;
                n_nearly_in_range[map_idx] = n_nearly_in_range_block ;
            }
        }
    }
}

///// generates a binomial Poisson cardinality distribution for the in-range features.
//__global__ void
//separateCardinalityKernel( Gaussian2D *features, int* map_offsets,
//                           REAL* cn_inrange)
//{
//    int n = threadIdx.x ;
//    int map_idx = blockIdx.x ;
//    int n_features = map_offsets[map_idx+1] - map_offsets[map_idx] ;
//    int feature_idx = map_offsets[map_idx] + n ;
//    REAL* cn_shared = (REAL*)shmem ;
//    REAL* weights = (REAL*)&cn_shared[dev_config.maxCardinality+1] ;

//    // compute product of weights
//    REAL val = 0 ;
//    if ( n < n_features )
//    {
//        val = log(features[ feature_idx ].weight) ;
//    }
//    sumByReduction( weights, val, n ) ;
//    REAL log_alpha = weights[0] ;
//    __syncthreads() ;

//    // load the polynomial roots into shared memory
//    if ( n < n_features )
//    {
//        weights[n] = (1-features[feature_idx].weight)/features[feature_idx].weight ;
//    }
//    else
//    {
//        weights[n] = 0 ;
//    }

//    // compute full cn using recursive algorithm
//    cn_shared[n+1] = 0 ;
//    int cn_offset = map_idx*(dev_config.maxCardinality+1) ;
//    if ( n == 0 )
//    {
//        cn_shared[0] = 1 ;
//    }
//    __syncthreads() ;
//    for ( int m = 0 ; m < n_features ; m++ )
//    {
//        REAL tmp1 = cn_shared[n+1] ;
//        REAL tmp2 = cn_shared[n] ;
//        __syncthreads() ;
//        if ( n < m+1 )
//            cn_shared[n+1] = tmp1 - weights[m]*tmp2 ;
//        __syncthreads() ;
//    }
//    if ( n <= n_features )
//    {
//        int idx = cn_offset + (n_features - n) ;
//        cn_inrange[idx] = safeLog(fabs(cn_shared[n]))
//                + log_alpha ;
//    }
//    else
//    {
//        cn_inrange[cn_offset+n] = LOG0 ;
//    }
//}

///// compute partially updated weights and updated means & covariances
///**
//  \param features Array of all Gaussians from all particles concatenated together
//  \param map_sizes Integer array indicating the number of features per particle.
//  \param n_particles Number of particles
//  \param n_measurements Number of measurements
//  \param poses Array of particle poses
//  \param w_partial Array of partially updated weights computed by kernel
//  */
//__global__ void
//cphdPreUpdateKernel(Gaussian2D *features, int* map_offsets,
//        int n_particles, int n_measurements, ConstantVelocityState* poses,
//        Gaussian2D* updated_features, REAL* w_partial, REAL* qdw )

//{
//    int tid = threadIdx.x + blockIdx.x*blockDim.x ;
//    int n_total = (n_measurements+1)*map_offsets[n_particles] ;
//    if ( tid >= n_total)
//        return ;
//    int map_idx = 0 ;
//    while ( map_offsets[map_idx]*(n_measurements+1) <= tid )
//    {
//        map_idx++ ;
//    }
//    map_idx-- ;
//    int n_features = map_offsets[map_idx+1] - map_offsets[map_idx] ;
//    int offset = map_offsets[map_idx]*(n_measurements+1) ;
//    int feature_idx = floor( (float)(tid-offset)/(n_measurements) ) ;

//    if ( feature_idx >= n_features ) // non-detect thread
//    {
//        int predict_idx = tid - n_features*n_measurements - offset
//                + map_offsets[map_idx] ;
//        updated_features[tid] = features[predict_idx] ;
//    }
//    else if ( tid < n_total ) // update thread
//    {
//        int z_idx = tid - feature_idx*n_measurements - offset ;

//        Gaussian2D feature = features[map_offsets[map_idx]+feature_idx] ;
//        Gaussian2D updated_feature ;
//        RangeBearingMeasurement z = Z[z_idx] ;
//        RangeBearingMeasurement z_predict ;
//        ConstantVelocityState pose = poses[map_idx] ;
//        REAL K[4] = {0,0,0,0} ;
//        REAL sigmaInv[4] = {0,0,0,0} ;
//        REAL covUpdate[4] = {0,0,0,0} ;
//        REAL featurePd = 0 ;
//        REAL detSigma = 0 ;

//        computePreUpdateComponents( pose, feature, K, covUpdate,
//                                    &detSigma, sigmaInv, &featurePd,
//                                    &z_predict ) ;

//        // innovation
//        REAL innov[2] = {0,0} ;
//        innov[0] = z.range - z_predict.range ;
//        innov[1] = wrapAngle(z.bearing - z_predict.bearing) ;

//        // updated mean
//        updated_feature.mean[0] = feature.mean[0] + K[0]*innov[0] + K[2]*innov[1] ;
//        updated_feature.mean[1] = feature.mean[1] + K[1]*innov[0] + K[3]*innov[1] ;

//        // updated covariances
//        updated_feature.cov[0] = covUpdate[0] ;
//        updated_feature.cov[1] = covUpdate[1] ;
//        updated_feature.cov[2] = covUpdate[2] ;
//        updated_feature.cov[3] = covUpdate[3] ;

//        // single-object likelihood
//        REAL dist = innov[0]*innov[0]*sigmaInv[0] +
//                innov[0]*innov[1]*(sigmaInv[1] + sigmaInv[2]) +
//                innov[1]*innov[1]*sigmaInv[3] ;

//        // partially updated weight
//        updated_feature.weight = safeLog(featurePd) + safeLog(feature.weight)
//                - 0.5*dist- safeLog(2*M_PI) - 0.5*safeLog(detSigma) ;

//        updated_features[tid] = updated_feature ;

//        int w_idx = map_offsets[map_idx]*n_measurements ;
//        w_idx += feature_idx*n_measurements + z_idx ;
//        w_partial[w_idx] = updated_feature.weight ;

//        if ( z_idx == 0 )
//        {
//            offset = map_offsets[map_idx] ;
//            qdw[offset+feature_idx] = safeLog(1-featurePd) + safeLog(feature.weight) ;
//        }
//    }
//}

///// computes the elementary symmetric polynomial coefficients
///**
//  This kernel produces the coefficients of the elementary symmetric function
//  for the CPHD update

//  \param w_partial Array of partially updated weights
//  \param map_sizes Number of features per particle
//  \param n_measurements Number of measurements
//  \param esf Array of ESF coefficients computed by kernel
//  \param esfd Array of ESF coefficients, with each measurement omitted
//  */
//__global__ void
//computeEsfKernel( REAL* w_partial, int* map_offsets, int n_measurements,
//                  REAL* esf, REAL* esfd )
//{
//    REAL* lambda = (REAL*)shmem ;
//    REAL* esf_shared = (REAL*)&lambda[n_measurements] ;

//    // determine indexing offsets
//    int tid = threadIdx.x ;
//    int map_idx = blockIdx.x ;
//    int block_offset = n_measurements*map_offsets[map_idx] ;
//    int n_features = map_offsets[map_idx+1] - map_offsets[map_idx] ;

//    // compute log lambda
//    lambda[tid] = 0 ;
//    int idx = block_offset + tid ;
//    REAL max_val = -FLT_MAX ;
//    for ( int j = 0 ; j < n_features ; j++)
//    {
//        REAL tmp = w_partial[idx] ;
//        REAL tmp_max = fmax(tmp,max_val) ;
//        lambda[tid] = exp( max_val - tmp_max )*lambda[tid]
//                + exp( tmp - tmp_max ) ;
//        max_val = tmp_max ;
//        idx += n_measurements ;
//    }
//    lambda[tid] = safeLog(lambda[tid]) + max_val
//            + safeLog(dev_config.clutterRate)
//            - safeLog(dev_config.clutterDensity) ;
//    __syncthreads() ;

//    // compute full esf using recursive algorithm
//    esf_shared[tid+1] = 0 ;
//    int esf_offset = map_idx*(n_measurements+1) ;
//    if ( tid == 0 )
//    {
//        esf_shared[0] = 1 ;
//        esf[esf_offset] = 0 ;
//    }
//    __syncthreads() ;
//    for ( int m = 0 ; m < n_measurements ; m++ )
//    {
//        REAL tmp1 = esf_shared[tid+1] ;
//        REAL tmp2 = esf_shared[tid] ;
//        __syncthreads() ;
//        if ( tid < m+1 )
//        {
////            REAL tmp_sum ;
////            max_val = fmax(tmp1, lambda[m]+tmp2) ;
////            tmp_sum = exp(tmp1-max_val) + exp(lambda[m]+tmp2-max_val) ;
////            esf_shared[tid+1] = safeLog( fabs(tmp_sum) ) + max_val ;
//            esf_shared[tid+1] = tmp1 - exp(lambda[m])*tmp2 ;
//        }
//        __syncthreads() ;
//    }
//    esf[esf_offset+tid+1] = log(fabs(esf_shared[tid+1])) ;

//    // compute esf's for detection terms
//    for ( int m = 0 ; m < n_measurements ; m++ )
//    {
//        int esfd_offset = n_measurements*n_measurements*map_idx + m*n_measurements ;
////        esf_shared[tid+1] = LOG0 ;
//        esf_shared[tid+1] = 0 ;
//        if ( tid == 0 )
//        {
////            esf_shared[0] = 0 ;
////            esfd[esfd_offset] = 0 ;
//            esf_shared[0] = 1 ;
//            esfd[esfd_offset] = 0 ;
//        }
//        __syncthreads() ;
//        int k = 0 ;
//        for ( int n = 0 ; n < n_measurements ; n++ )
//        {
//            REAL tmp1 = esf_shared[tid+1] ;
//            REAL tmp2 = esf_shared[tid] ;
//            __syncthreads() ;
//            if ( n != m )
//            {
//                if ( tid < k+1 )
//                {
////                    REAL tmp_sum ;
////                    max_val = fmax(tmp1,lambda[n]+tmp2) ;
////                    tmp_sum = exp(tmp1-max_val) - exp(lambda[n]+tmp2-max_val) ;
////                    esf_shared[tid+1] = safeLog( fabs(tmp_sum) ) + max_val ;
//                    esf_shared[tid+1] = tmp1 - exp(lambda[n])*tmp2 ;
//                }
//                k++ ;
//            }
//            __syncthreads() ;
//        }
//        if ( tid < (n_measurements-1) )
//            esfd[esfd_offset+tid+1] = log(fabs(esf_shared[tid+1])) ;
//    }
//}

///// compute the multi-object likelihoods for the CPHD update
///**
//  This kernel computes the terms denoted as Psi in Vo's Analytic CPHD paper, and
//  their inner products with the predicted cardinality distribution. It also
//  produces the updated cardinality
//  */
//__global__ void
//computePsiKernel( Gaussian2D* features_predict, REAL* cn_predict, REAL* esf,
//                  REAL* esfd, int* map_offsets,
//                  int n_measurements, REAL* qdw, REAL* dev_factorial,
//                  REAL* dev_C, REAL* dev_cn_clutter, REAL* cn_update,
//                  REAL* innerprod_psi0, REAL* innerprod_psi1,
//                  REAL* innerprod_psi1d )
//{
//    int n = threadIdx.x ;
//    REAL psi0 = 0 ;
//    REAL psi1 = 0 ;
//    int map_idx = blockIdx.x ;
//    int cn_offset = (dev_config.maxCardinality+1)*map_idx ;
//    int esf_offset = (n_measurements+1)*map_idx ;
//    int stop_idx = 0 ;
//    REAL max_val0 = 0 ;
//    REAL max_val1 = 0 ;
//    REAL* shdata = (REAL*)shmem ;

//    // compute the (log) inner product < q_D, w >
//    int map_offset = map_offsets[map_idx] ;
//    int n_features = map_offsets[map_idx+1] - map_offsets[map_idx] ;
//    REAL innerprod_qdw = 0 ;
//    max_val0 = qdw[map_offset] ;
//    for ( int j = 0 ; j < n_features ; j+=blockDim.x )
//    {
//        REAL val = -FLT_MAX ;
//        if ( j+n < n_features )
//            val = qdw[map_offset+j+n] ;
//        maxByReduction(shdata,val,n) ;
//        max_val0 = fmax(max_val0,shdata[0]) ;
//        __syncthreads() ;
//    }
//    for ( int j = 0 ; j < n_features ; j+= blockDim.x )
//    {
//        REAL val = 0 ;
//        if ( (j+n) < n_features )
//            val = exp(qdw[map_offset+j+n]-max_val0) ;
//        sumByReduction( shdata, val, n ) ;
//        innerprod_qdw += shdata[0] ;
//        __syncthreads() ;
//    }
//    innerprod_qdw = safeLog(innerprod_qdw) + max_val0 ;

//    // compute the (log) inner product < 1, w >
//    REAL wsum = 0 ;
//    for ( int j = 0 ; j < n_features ; j += blockDim.x )
//    {
//        REAL val = 0 ;
//        if ( (j+n) < n_features )
//            val = features_predict[map_offset+j+n].weight ;
//        sumByReduction( shdata, val, n );
//        wsum += shdata[0] ;
//        __syncthreads() ;
//    }
//    wsum = safeLog(wsum) ;

//    // compute (log) PSI0(n) and PSI1(n), using log-sum-exp
//    max_val0 = -FLT_MAX ;
//    max_val1 = -FLT_MAX ;
//    stop_idx = min(n,n_measurements) ;
//    for ( int j = 0 ; j <= stop_idx ; j++ )
//    {
//        // PSI0
//        REAL p_coeff = dev_C[n+j*(dev_config.maxCardinality+1)]
//                + dev_factorial[j] ;
//        REAL aux = dev_factorial[n_measurements-j]
//                + dev_cn_clutter[n_measurements-j] + esf[esf_offset+ j]
//                - n*wsum ;
//        REAL tmp =  aux + p_coeff + (n-j)*innerprod_qdw ;

//        psi0 = exp(max_val0-fmax(max_val0,tmp))*psi0
//                + exp(tmp - fmax(max_val0,tmp) ) ;
//        max_val0 = fmax(max_val0,tmp) ;

//        // PSI1
//        p_coeff = dev_C[n+(j+1)*(dev_config.maxCardinality+1)]
//                + dev_factorial[j+1] ;
//        tmp = aux + p_coeff + (n-(j+1))*innerprod_qdw  ;
//        psi1 = exp(max_val1-fmax(max_val1,tmp))*psi1
//                + exp(tmp - fmax(max_val1,tmp) ) ;
//        max_val1 = fmax(max_val1,tmp) ;
//    }
//    psi0 = safeLog(psi0) + max_val0 ;
//    psi1 = safeLog(psi1) + max_val1 ;

//    // (log) inner product of PSI0 and predicted cardinality distribution, using
//    // log-sum-exp trick
//    REAL val = psi0 + cn_predict[cn_offset+n] ;
//    maxByReduction( shdata, val, n ) ;
//    max_val0 = shdata[0] ;
//    __syncthreads() ;
//    sumByReduction( shdata, exp(val-max_val0), n ) ;
//    if ( n==0 )
//        innerprod_psi0[map_idx] = safeLog(shdata[0]) + max_val0 ;


//    // (log) inner product of PSI1 and predicted cardinality distribution, using
//    // log-sum-exp trick
//    val = psi1 + cn_predict[cn_offset+n] ;
//    maxByReduction( shdata, psi1+cn_predict[cn_offset+n], n ) ;
////	shdata[n] = psi1+cn_predict[cn_offset+n] ;
//    max_val1 = shdata[0] ;
//    __syncthreads() ;
//    sumByReduction( shdata, exp( val - max_val1 ), n ) ;
//    if ( n == 0 )
//        innerprod_psi1[map_idx] = safeLog(shdata[0]) + max_val1 ;
////	__syncthreads() ;

//    // PSI1 detection terms
//    stop_idx = min(n_measurements - 1, n) ;
//    for ( int m = 0 ; m < n_measurements ; m++ )
//    {
//        int esfd_offset = map_idx * n_measurements * n_measurements
//                + m*n_measurements ;
//        REAL psi1d = 0 ;
//        max_val1 = -FLT_MAX ;
//        for ( int j = 0 ; j <= stop_idx ; j++ )
//        {
//            REAL p_coeff = dev_C[n+(j+1)*(dev_config.maxCardinality+1)]
//                    + dev_factorial[j+1] ;
//            REAL aux = dev_factorial[n_measurements-1-j]
//                + dev_cn_clutter[n_measurements-1-j] + esfd[esfd_offset+ j]
//                - n*wsum ;
//            REAL tmp = aux + p_coeff + (n-(j+1))*innerprod_qdw ;
//            psi1d = exp(max_val1-fmax(max_val1,tmp))*psi1d
//                    + exp(tmp - fmax(max_val1,tmp) ) ;
//            max_val1 = fmax(max_val1,tmp) ;
//        }
//        psi1d = safeLog(psi1d) + max_val1 ;
//        val = psi1d + cn_predict[cn_offset+n] ;
//        maxByReduction( shdata, val, n ) ;
//        max_val1 = shdata[0] ;
//        __syncthreads() ;
//        sumByReduction( shdata, exp(val-max_val1), n ) ;
//        if ( n == 0 )
//            innerprod_psi1d[map_idx*n_measurements+m] = safeLog(shdata[0]) + max_val1 ;
//        __syncthreads() ;
//    }

//    // compute log updated cardinality
//    cn_update[cn_offset+n] = cn_predict[cn_offset+n] + psi0
//            - innerprod_psi0[map_idx] ;
//}

///// perform the gaussian mixture CPHD weight update
///**
//  This kernel takes the results produced by the previous three kernels in the
//  CPHD pipeline (PreUpdate, ComputeEsf, and ComputePsi) and applies them to
//  update the weights of the Gaussian Mixture as in Vo's paper

//  Kernel organization: One thread block per particle. Each thread updates all
//  the features for one measurement.
//  */
//__global__ void
//cphdUpdateKernel( int* map_offsets, int n_measurements,
//                  REAL* innerprod_psi0, REAL* innerprod_psi1,
//                  REAL* innerprod_psi1d, bool* merged_flags,
//                  Gaussian2D* updated_features )
//{
//    int z_idx = threadIdx.x ;
//    int map_idx = blockIdx.x ;
//    int offset = (n_measurements+1)*map_offsets[map_idx] ;
//    int n_features = map_offsets[map_idx+1] - map_offsets[map_idx] ;

//    // detection update
//    REAL psi1d = innerprod_psi1d[n_measurements*map_idx+z_idx] ;
//    for ( int j = 0 ; j < n_features ; j++ )
//    {
//        REAL tmp = updated_features[offset+z_idx].weight
//                + psi1d - innerprod_psi0[map_idx] + safeLog(dev_config.clutterRate)
//                - safeLog(dev_config.clutterDensity) ;
//        updated_features[offset+z_idx].weight = exp(tmp) ;
//        if ( exp(tmp) >= dev_config.minFeatureWeight )
//            merged_flags[offset + z_idx] = false ;
//        else
//            merged_flags[offset + z_idx] = true ;
//        offset += n_measurements ;
//    }

//    // non-detection updates
//    for ( int j = 0 ; j < n_features ; j += blockDim.x )
//    {
//        if ( j+z_idx < n_features )
//        {
//            int nondetect_idx = offset + j + z_idx ;
//            REAL tmp = safeLog(updated_features[nondetect_idx].weight)
//                    + innerprod_psi1[map_idx] - innerprod_psi0[map_idx]
//                    + safeLog(1-dev_config.pd) ;
//            updated_features[nondetect_idx].weight = exp(tmp) ;
//            if ( exp(tmp) >= dev_config.minFeatureWeight )
//                merged_flags[nondetect_idx] = false ;
//            else
//                merged_flags[nondetect_idx] = true ;
//        }
//    }
//}

__global__ void
preUpdateSynthKernel(ConstantVelocityState* poses,
                     int* pose_indices,
                     Gaussian2D* features_predict,
                     REAL* features_pd,
                     int n_features, int n_measure,
                     REAL* likelihoods,
                     Gaussian2D* features_preupdate){
    int tid = blockIdx.x*blockDim.x + threadIdx.x ;
    for ( int i = tid ; i < n_features ; i+= gridDim.x*blockDim.x){
        // get vehicle pose
        ConstantVelocityState pose = poses[pose_indices[i]] ;

        // get predicted feature
        Gaussian2D feature_predict = features_predict[i] ;

        // predicted measurement
        REAL dx = feature_predict.mean[0] - pose.px ;
        REAL dy = feature_predict.mean[1] - pose.py ;
        REAL r2 = dx*dx + dy*dy ;
        REAL r = sqrt(r2) ;
        REAL bearing = wrapAngle(atan2f(dy,dx) - pose.ptheta) ;

        // probability of detection
        REAL feature_pd = 0 ;
        if ( r <= dev_config.maxRange && fabsf(bearing) <= dev_config.maxBearing )
            feature_pd = dev_config.pd ;
        features_pd[i] = feature_pd ;

        // measurement jacobian wrt feature
        REAL J[4] ;
        J[0] = dx/r ;
        J[2] = dy/r ;
        J[1] = -dy/r2 ;
        J[3] = dx/r2 ;

        REAL* P = feature_predict.cov ;

        // BEGIN Maple-Generated expressions
        // innovation covariance
        REAL sigma[4] ;
        sigma[0] = (P[0] * J[0] + J[2] * P[1]) * J[0] + (J[0] * P[2] + P[3] * J[2]) * J[2] + pow(dev_config.stdRange,2) ;
        sigma[1] = (P[0] * J[1] + J[3] * P[1]) * J[0] + (J[1] * P[2] + P[3] * J[3]) * J[2];
        sigma[2] = (P[0] * J[0] + J[2] * P[1]) * J[1] + (J[0] * P[2] + P[3] * J[2]) * J[3];
        sigma[3] = (P[0] * J[1] + J[3] * P[1]) * J[1] + (J[1] * P[2] + P[3] * J[3]) * J[3] + pow(dev_config.stdBearing,2) ;

        // enforce symmetry
        sigma[1] = (sigma[1]+sigma[2])/2 ;
        sigma[2] = sigma[1] ;

        REAL det_sigma = sigma[0]*sigma[3] - sigma[1]*sigma[2] ;

        // inverse of sigma
        REAL S[4] ;
        S[0] = sigma[3]/(det_sigma) ;
        S[1] = -sigma[1]/(det_sigma) ;
        S[2] = -sigma[2]/(det_sigma) ;
        S[3] = sigma[0]/(det_sigma) ;

        // Kalman gain
        REAL K[4] ;
        K[0] = S[0]*(P[0]*J[0] + P[2]*J[2]) + S[1]*(P[0]*J[1] + P[2]*J[3]) ;
        K[1] = S[0]*(P[1]*J[0] + P[3]*J[2]) + S[1]*(P[1]*J[1] + P[3]*J[3]) ;
        K[2] = S[2]*(P[0]*J[0] + P[2]*J[2]) + S[3]*(P[0]*J[1] + P[2]*J[3]) ;
        K[3] = S[2]*(P[1]*J[0] + P[3]*J[2]) + S[3]*(P[1]*J[1] + P[3]*J[3]) ;

        REAL cov_update[4] ;
        cov_update[0] = ((1 - K[0] * J[0] - K[2] * J[1]) * P[0] + (-K[0] * J[2] - K[2] * J[3]) * P[1]) * (1 - K[0] * J[0] - K[2] * J[1]) + ((1 - K[0] * J[0] - K[2] * J[1]) * P[2] + (-K[0] * J[2] - K[2] * J[3]) * P[3]) * (-K[0] * J[2] - K[2] * J[3]) + pow(K[0], 2) * dev_config.stdRange*dev_config.stdRange + pow(K[2], 2) * dev_config.stdBearing*dev_config.stdBearing;
        cov_update[2] = ((1 - K[0] * J[0] - K[2] * J[1]) * P[0] + (-K[0] * J[2] - K[2] * J[3]) * P[1]) * (-K[1] * J[0] - K[3] * J[1]) + ((1 - K[0] * J[0] - K[2] * J[1]) * P[2] + (-K[0] * J[2] - K[2] * J[3]) * P[3]) * (1 - K[1] * J[2] - K[3] * J[3]) + K[0] * dev_config.stdRange*dev_config.stdRange * K[1] + K[2] * dev_config.stdBearing*dev_config.stdBearing * K[3];
        cov_update[1] = ((-K[1] * J[0] - K[3] * J[1]) * P[0] + (1 - K[1] * J[2] - K[3] * J[3]) * P[1]) * (1 - K[0] * J[0] - K[2] * J[1]) + ((-K[1] * J[0] - K[3] * J[1]) * P[2] + (1 - K[1] * J[2] - K[3] * J[3]) * P[3]) * (-K[0] * J[2] - K[2] * J[3]) + K[0] * dev_config.stdRange*dev_config.stdRange * K[1] + K[2] * dev_config.stdBearing*dev_config.stdBearing * K[3];
        cov_update[3] = ((-K[1] * J[0] - K[3] * J[1]) * P[0] + (1 - K[1] * J[2] - K[3] * J[3]) * P[1]) * (-K[1] * J[0] - K[3] * J[1]) + ((-K[1] * J[0] - K[3] * J[1]) * P[2] + (1 - K[1] * J[2] - K[3] * J[3]) * P[3]) * (1 - K[1] * J[2] - K[3] * J[3]) + pow(K[1], 2) * dev_config.stdRange*dev_config.stdRange + pow(K[3], 2) * dev_config.stdBearing*dev_config.stdBearing;

        REAL innov[2] ;
        REAL dist = 0 ;
        for ( int m = 0 ; m < n_measure ; m++ )
        {
            int idx = m*n_features + i ;
            innov[0] = Z[m].range - r ;
            innov[1] = wrapAngle(Z[m].bearing - bearing) ;
            features_preupdate[idx].mean[0] = feature_predict.mean[0] + K[0]*innov[0] + K[2]*innov[1] ;
            features_preupdate[idx].mean[1] = feature_predict.mean[1] + K[1]*innov[0] + K[3]*innov[1] ;
            for ( int n = 0 ; n < 4 ; n++ )
                features_preupdate[idx].cov[n] = cov_update[n] ;
            // compute single object likelihood
            dist = innov[0]*innov[0]*S[0] +
                    innov[0]*innov[1]*(S[1] + S[2]) +
                    innov[1]*innov[1]*S[3] ;
            REAL g = - 0.5*dist - safeLog(2*M_PI) - 0.5*safeLog(det_sigma) ;
            likelihoods[idx] = exp(g) ;
            if(Z[m].label==STATIC_MEASUREMENT || !dev_config.labeledMeasurements)
            {
                // partially update weight (log-transformed)
                features_preupdate[idx].weight = safeLog(feature_pd)
                        + safeLog(feature_predict.weight) + g ;
            }
            else
            {
                features_preupdate[idx].weight = safeLog(0) ;
            }
        }
    }
}

__global__ void
preUpdateSynthKernel(ConstantVelocityState* poses,
                     int* pose_indices,
                     Gaussian4D* features_predict,
                     REAL* features_pd,
                     int n_features, int n_measure,
                     Gaussian4D* features_preupdate){
    int tid = blockIdx.x*blockDim.x + threadIdx.x ;
    for ( int i = tid ; i < n_features ; i+= gridDim.x*blockDim.x){
        // get vehicle pose
        ConstantVelocityState pose = poses[pose_indices[i]] ;

        // get predicted feature
        Gaussian4D feature_predict = features_predict[i] ;

        // predicted measurement
        REAL dx = feature_predict.mean[0] - pose.px ;
        REAL dy = feature_predict.mean[1] - pose.py ;
        REAL r2 = dx*dx + dy*dy ;
        REAL r = sqrt(r2) ;
        REAL bearing = wrapAngle(atan2f(dy,dx) - pose.ptheta) ;

        // probability of detection
        REAL feature_pd = 0 ;
        if ( r <= dev_config.maxRange && fabsf(bearing) <= dev_config.maxBearing )
            feature_pd = dev_config.pd ;
        features_pd[i] = feature_pd ;

        // measurement jacobian wrt feature
        REAL J[4] ;
        J[0] = dx/r ;
        J[2] = dy/r ;
        J[1] = -dy/r2 ;
        J[3] = dx/r2 ;

        REAL* P = feature_predict.cov ;

        // BEGIN Maple-Generated expressions
        // innovation covariance
        REAL sigma[4] ;
        REAL var_range = pow(dev_config.stdRange,2) ;
        REAL var_bearing = pow(dev_config.stdBearing,2) ;
        sigma[0] = J[0] * (P[0] * J[0] + P[4] * J[2]) + J[2] * (P[1] * J[0] + P[5] * J[2]) + var_range;
        sigma[1] = J[1] * (P[0] * J[0] + P[4] * J[2]) + J[3] * (P[1] * J[0] + P[5] * J[2]);
        sigma[2] = J[0] * (P[0] * J[1] + P[4] * J[3]) + J[2] * (P[1] * J[1] + P[5] * J[3]);
        sigma[3] = J[1] * (P[0] * J[1] + P[4] * J[3]) + J[3] * (P[1] * J[1] + P[5] * J[3]) + var_bearing;

        // enforce symmetry
        sigma[1] = (sigma[1]+sigma[2])/2 ;
        sigma[2] = sigma[1] ;
    //			makePositiveDefinite(sigma) ;

        REAL det_sigma = sigma[0]*sigma[3] - sigma[1]*sigma[2] ;

        REAL S[4] ;
        S[0] = sigma[3]/(det_sigma) ;
        S[1] = -sigma[1]/(det_sigma) ;
        S[2] = -sigma[2]/(det_sigma) ;
        S[3] = sigma[0]/(det_sigma) ;

        // Kalman gain
        REAL K[8] ;
        K[0] = P[0] * (J[0] * S[0] + J[1] * S[1])
                + P[4] * (J[2] * S[0] + J[3] * S[1]);
        K[1] = P[1] * (J[0] * S[0] + J[1] * S[1])
                + P[5] * (J[2] * S[0] + J[3] * S[1]);
        K[2] = P[2] * (J[0] * S[0] + J[1] * S[1])
                + P[6] * (J[2] * S[0] + J[3] * S[1]);
        K[3] = P[3] * (J[0] * S[0] + J[1] * S[1])
                + P[7] * (J[2] * S[0] + J[3] * S[1]);
        K[4] = P[0] * (J[0] * S[2] + J[1] * S[3])
                + P[4] * (J[2] * S[2] + J[3] * S[3]);
        K[5] = P[1] * (J[0] * S[2] + J[1] * S[3])
                + P[5] * (J[2] * S[2] + J[3] * S[3]);
        K[6] = P[2] * (J[0] * S[2] + J[1] * S[3])
                + P[6] * (J[2] * S[2] + J[3] * S[3]);
        K[7] = P[3] * (J[0] * S[2] + J[1] * S[3])
                + P[7] * (J[2] * S[2] + J[3] * S[3]);

        // Updated covariance (Joseph Form)
        REAL cov_update[16] ;
        cov_update[0] = (1 - K[0] * J[0] - K[4] * J[1]) * (P[0] * (1 - K[0] * J[0] - K[4] * J[1]) + P[4] * (-K[0] * J[2] - K[4] * J[3])) + (-K[0] * J[2] - K[4] * J[3]) * (P[1] * (1 - K[0] * J[0] - K[4] * J[1]) + P[5] * (-K[0] * J[2] - K[4] * J[3])) + var_range *  pow( K[0],  2) + var_bearing *  pow( K[4],  2);
        cov_update[1] = (-K[1] * J[0] - K[5] * J[1]) * (P[0] * (1 - K[0] * J[0] - K[4] * J[1]) + P[4] * (-K[0] * J[2] - K[4] * J[3])) + (1 - K[1] * J[2] - K[5] * J[3]) * (P[1] * (1 - K[0] * J[0] - K[4] * J[1]) + P[5] * (-K[0] * J[2] - K[4] * J[3])) + K[0] * var_range * K[1] + K[4] * var_bearing * K[5];
        cov_update[2] = (-K[2] * J[0] - K[6] * J[1]) * (P[0] * (1 - K[0] * J[0] - K[4] * J[1]) + P[4] * (-K[0] * J[2] - K[4] * J[3])) + (-K[2] * J[2] - K[6] * J[3]) * (P[1] * (1 - K[0] * J[0] - K[4] * J[1]) + P[5] * (-K[0] * J[2] - K[4] * J[3])) + P[2] * (1 - K[0] * J[0] - K[4] * J[1]) + P[6] * (-K[0] * J[2] - K[4] * J[3]) + K[0] * var_range * K[2] + K[4] * var_bearing * K[6];
        cov_update[3] = (-K[3] * J[0] - K[7] * J[1]) * (P[0] * (1 - K[0] * J[0] - K[4] * J[1]) + P[4] * (-K[0] * J[2] - K[4] * J[3])) + (-K[3] * J[2] - K[7] * J[3]) * (P[1] * (1 - K[0] * J[0] - K[4] * J[1]) + P[5] * (-K[0] * J[2] - K[4] * J[3])) + P[3] * (1 - K[0] * J[0] - K[4] * J[1]) + P[7] * (-K[0] * J[2] - K[4] * J[3]) + K[0] * var_range * K[3] + K[4] * var_bearing * K[7];
        cov_update[4] = (1 - K[0] * J[0] - K[4] * J[1]) * (P[0] * (-K[1] * J[0] - K[5] * J[1]) + P[4] * (1 - K[1] * J[2] - K[5] * J[3])) + (-K[0] * J[2] - K[4] * J[3]) * (P[1] * (-K[1] * J[0] - K[5] * J[1]) + P[5] * (1 - K[1] * J[2] - K[5] * J[3])) + K[0] * var_range * K[1] + K[4] * var_bearing * K[5];
        cov_update[5] = (-K[1] * J[0] - K[5] * J[1]) * (P[0] * (-K[1] * J[0] - K[5] * J[1]) + P[4] * (1 - K[1] * J[2] - K[5] * J[3])) + (1 - K[1] * J[2] - K[5] * J[3]) * (P[1] * (-K[1] * J[0] - K[5] * J[1]) + P[5] * (1 - K[1] * J[2] - K[5] * J[3])) + var_range *  pow( K[1],  2) + var_bearing *  pow( K[5],  2);
        cov_update[6] = (-K[2] * J[0] - K[6] * J[1]) * (P[0] * (-K[1] * J[0] - K[5] * J[1]) + P[4] * (1 - K[1] * J[2] - K[5] * J[3])) + (-K[2] * J[2] - K[6] * J[3]) * (P[1] * (-K[1] * J[0] - K[5] * J[1]) + P[5] * (1 - K[1] * J[2] - K[5] * J[3])) + P[2] * (-K[1] * J[0] - K[5] * J[1]) + P[6] * (1 - K[1] * J[2] - K[5] * J[3]) + K[1] * var_range * K[2] + K[5] * var_bearing * K[6];
        cov_update[7] = (-K[3] * J[0] - K[7] * J[1]) * (P[0] * (-K[1] * J[0] - K[5] * J[1]) + P[4] * (1 - K[1] * J[2] - K[5] * J[3])) + (-K[3] * J[2] - K[7] * J[3]) * (P[1] * (-K[1] * J[0] - K[5] * J[1]) + P[5] * (1 - K[1] * J[2] - K[5] * J[3])) + P[3] * (-K[1] * J[0] - K[5] * J[1]) + P[7] * (1 - K[1] * J[2] - K[5] * J[3]) + K[1] * var_range * K[3] + K[5] * var_bearing * K[7];
        cov_update[8] = (1 - K[0] * J[0] - K[4] * J[1]) * (P[0] * (-K[2] * J[0] - K[6] * J[1]) + P[4] * (-K[2] * J[2] - K[6] * J[3]) + P[8]) + (-K[0] * J[2] - K[4] * J[3]) * (P[1] * (-K[2] * J[0] - K[6] * J[1]) + P[5] * (-K[2] * J[2] - K[6] * J[3]) + P[9]) + K[0] * var_range * K[2] + K[4] * var_bearing * K[6];
        cov_update[9] = (-K[1] * J[0] - K[5] * J[1]) * (P[0] * (-K[2] * J[0] - K[6] * J[1]) + P[4] * (-K[2] * J[2] - K[6] * J[3]) + P[8]) + (1 - K[1] * J[2] - K[5] * J[3]) * (P[1] * (-K[2] * J[0] - K[6] * J[1]) + P[5] * (-K[2] * J[2] - K[6] * J[3]) + P[9]) + K[1] * var_range * K[2] + K[5] * var_bearing * K[6];
        cov_update[10] = (-K[2] * J[0] - K[6] * J[1]) * (P[0] * (-K[2] * J[0] - K[6] * J[1]) + P[4] * (-K[2] * J[2] - K[6] * J[3]) + P[8]) + (-K[2] * J[2] - K[6] * J[3]) * (P[1] * (-K[2] * J[0] - K[6] * J[1]) + P[5] * (-K[2] * J[2] - K[6] * J[3]) + P[9]) + P[2] * (-K[2] * J[0] - K[6] * J[1]) + P[6] * (-K[2] * J[2] - K[6] * J[3]) + P[10] + var_range *  pow( K[2],  2) + var_bearing *  pow( K[6],  2);
        cov_update[11] = (-K[3] * J[0] - K[7] * J[1]) * (P[0] * (-K[2] * J[0] - K[6] * J[1]) + P[4] * (-K[2] * J[2] - K[6] * J[3]) + P[8]) + (-K[3] * J[2] - K[7] * J[3]) * (P[1] * (-K[2] * J[0] - K[6] * J[1]) + P[5] * (-K[2] * J[2] - K[6] * J[3]) + P[9]) + P[3] * (-K[2] * J[0] - K[6] * J[1]) + P[7] * (-K[2] * J[2] - K[6] * J[3]) + P[11] + K[2] * var_range * K[3] + K[6] * var_bearing * K[7];
        cov_update[12] = (1 - K[0] * J[0] - K[4] * J[1]) * (P[0] * (-K[3] * J[0] - K[7] * J[1]) + P[4] * (-K[3] * J[2] - K[7] * J[3]) + P[12]) + (-K[0] * J[2] - K[4] * J[3]) * (P[1] * (-K[3] * J[0] - K[7] * J[1]) + P[5] * (-K[3] * J[2] - K[7] * J[3]) + P[13]) + K[0] * var_range * K[3] + K[4] * var_bearing * K[7];
        cov_update[13] = (-K[1] * J[0] - K[5] * J[1]) * (P[0] * (-K[3] * J[0] - K[7] * J[1]) + P[4] * (-K[3] * J[2] - K[7] * J[3]) + P[12]) + (1 - K[1] * J[2] - K[5] * J[3]) * (P[1] * (-K[3] * J[0] - K[7] * J[1]) + P[5] * (-K[3] * J[2] - K[7] * J[3]) + P[13]) + K[1] * var_range * K[3] + K[5] * var_bearing * K[7];
        cov_update[14] = (-K[2] * J[0] - K[6] * J[1]) * (P[0] * (-K[3] * J[0] - K[7] * J[1]) + P[4] * (-K[3] * J[2] - K[7] * J[3]) + P[12]) + (-K[2] * J[2] - K[6] * J[3]) * (P[1] * (-K[3] * J[0] - K[7] * J[1]) + P[5] * (-K[3] * J[2] - K[7] * J[3]) + P[13]) + P[2] * (-K[3] * J[0] - K[7] * J[1]) + P[6] * (-K[3] * J[2] - K[7] * J[3]) + P[14] + K[2] * var_range * K[3] + K[6] * var_bearing * K[7];
        cov_update[15] = (-K[3] * J[0] - K[7] * J[1]) * (P[0] * (-K[3] * J[0] - K[7] * J[1]) + P[4] * (-K[3] * J[2] - K[7] * J[3]) + P[12]) + (-K[3] * J[2] - K[7] * J[3]) * (P[1] * (-K[3] * J[0] - K[7] * J[1]) + P[5] * (-K[3] * J[2] - K[7] * J[3]) + P[13]) + P[3] * (-K[3] * J[0] - K[7] * J[1]) + P[7] * (-K[3] * J[2] - K[7] * J[3]) + P[15] + var_range *  pow( K[3],  2) + var_bearing *  pow( K[7],  2);

        REAL innov[2] ;
        REAL dist = 0 ;
        for ( int m = 0 ; m < n_measure ; m++ )
        {
            int idx = m*n_features+i ;

            innov[0] = Z[m].range - r ;
            innov[1] = wrapAngle(Z[m].bearing - bearing) ;
            features_preupdate[idx].mean[0] = feature_predict.mean[0] + K[0]*innov[0] + K[4]*innov[1] ;
            features_preupdate[idx].mean[1] = feature_predict.mean[1] + K[1]*innov[0] + K[5]*innov[1] ;
            features_preupdate[idx].mean[2] = feature_predict.mean[2] + K[2]*innov[0] + K[6]*innov[1] ;
            features_preupdate[idx].mean[3] = feature_predict.mean[3] + K[3]*innov[0] + K[7]*innov[1] ;
            for ( int n = 0 ; n < 16 ; n++ )
                features_preupdate[idx].cov[n] = cov_update[n] ;
            // compute single object likelihood
            dist = innov[0]*innov[0]*S[0] +
                    innov[0]*innov[1]*(S[1] + S[2]) +
                    innov[1]*innov[1]*S[3] ;
            if(Z[m].label==DYNAMIC_MEASUREMENT || !dev_config.labeledMeasurements)
            {
                // partially update weight (log-transformed)
                features_preupdate[idx].weight = safeLog(feature_pd)
                        + safeLog(feature_predict.weight)
                        - 0.5*dist
                        - safeLog(2*M_PI)
                        - 0.5*safeLog(det_sigma) ;
            }
            else
            {
                features_preupdate[idx].weight = safeLog(0) ;
            }
        }
    }
}

/// perform the gaussian mixture PHD update
/**
  PHD update algorithm as in Vo & Ma 2006.
    \param inRangeFeatures Array of in-range Gaussians, with which the PHD
        update will be performed
    \param map_sizes_static Integer array of sizes of each particle's map
    \param n_measure Number of measurements
    \param poses Array of particle poses
    \param compatibleZ char array which will be computed by the kernel.
        Indicates which measurements have been found compatible with an existing
        gaussian.
    \param updated_features Stores the updated Gaussians computed by the kernel
    \param mergedFeatures Stores the post-merge updated Gaussians computed by
        the kernel.
    \param mergedSizes Stores the number of Gaussians left in each map after
        merging. This is required because the same amount of memory is allocated
        for both updated_features and mergedFeatures. The indexing boundaries for
        the maps will be the same, but only the first n gaussians after the
        boundary will be valid for the mergedFeatures array.
    \param mergedFlags Array of booleans used by the merging algorithm to keep
        track of which features have already be merged.
    \param particleWeights New particle weights after PHD update
  */
template <class GaussianType>
__global__ void
phdUpdateKernel(GaussianType* features_predict,
                REAL* featurePd,
                GaussianType* features_preupdate,
                GaussianType* features_birth,
                int* map_offsets,
                int n_particles, int n_measure,
                GaussianType* features_update,
                bool* merge_flags,
                REAL* particle_weights)
{
    // shared memory variables
    __shared__ REAL sdata[256] ;

    REAL particle_weight = 0 ;
    REAL cardinality_predict = 0 ;
    int update_offset = 0 ;
    int n_features = 0 ;
    int n_update = 0 ;
    int predict_offset = 0 ;
    int preupdate_offset = 0 ;
    int birth_offset = 0 ;


    // initialize variables
    int tid = threadIdx.x ;
    // pre-update variables
    GaussianType feature ;

    // update variables
    int preupdate_stride = map_offsets[n_particles] ;
    REAL w_partial = 0 ;
    int updateIdx = 0 ;

    // loop over particles
    for ( int map_idx = blockIdx.x ; map_idx < n_particles ; map_idx += gridDim.x )
    {
        // initialize map-specific variables
        predict_offset = map_offsets[map_idx] ;
        update_offset = predict_offset*(n_measure+1) +
                map_idx*n_measure ;
        preupdate_offset = predict_offset ;
        n_features = map_offsets[map_idx+1] - map_offsets[map_idx] ;
        n_update = (n_features)*(n_measure+1) + n_measure ;
        particle_weight = 0 ;
        cardinality_predict = 0.0 ;
        birth_offset = map_idx*n_measure ;

        // loop over predicted features + newborn features
        for ( int j = 0 ; j < (n_features+n_measure) ; j += blockDim.x )
        {
            int feature_idx = j + tid ;
            w_partial = 0 ;
            if ( feature_idx < n_features )
            {
                // persistent feature
                feature = features_predict[predict_offset+feature_idx] ;

                REAL pd = featurePd[predict_offset+feature_idx] ;

                // save non-detection term
                int idx_nondetect = update_offset
                        + feature_idx ;
                copy_gaussians(feature,features_update[idx_nondetect]) ;
                features_update[idx_nondetect].weight *= (1-pd) ;

                // save the detection terms
                for (int m = 0 ; m < n_measure ; m++){
                    int preupdate_idx = m*preupdate_stride +
                            preupdate_offset + feature_idx ;
                    int update_idx = update_offset + n_features +
                            m*n_features + feature_idx ;
                    copy_gaussians(features_preupdate[preupdate_idx],
                                   features_update[update_idx]) ;
                }

                w_partial = pd*feature.weight ;
            }
            else if (feature_idx < n_features+n_measure)
            {
                // newborn feature

                // find measurement corresponding to current thread
                int z_idx = feature_idx - n_features ;

                int idx_birth = update_offset + n_features
                        + n_measure*n_features + z_idx ;
                copy_gaussians(features_birth[birth_offset+z_idx],
                               features_update[idx_birth]) ;

                w_partial = dev_config.birthWeight ;
            }
            else
            {
                // thread does not correspond to a feature
                w_partial = 0 ;
            }

            // compute predicted cardinality
            sumByReduction(sdata, w_partial, tid) ;
            cardinality_predict += sdata[0] ;
            __syncthreads() ;
        }

//            cuPrintf("particle_weight=%f\n",particle_weight) ;
        // compute the weight normalizers
        for ( int i = 0 ; i < n_measure ; i++ )
        {
            REAL log_normalizer = 0 ;
            REAL val = 0 ;
            REAL sum = 0 ;
            GaussianType* ptr_update = features_update
                    + update_offset + n_features + i*n_features ;
//                cuPrintf("%f\n",features_update[0].weight) ;
            if (n_features > 0)
            {
                // find the maximum from all the log partial weights
                for ( int j = 0 ; j < (n_features) ; j += blockDim.x )
                {
                    int feature_idx = j + tid ;
                    if ( feature_idx < n_features )
                        val = exp(ptr_update[feature_idx].weight) ;
                    else
                        val = 0 ;
                    sumByReduction(sdata,val,tid);
                    sum += sdata[0] ;
                }

                // add clutter density and birth weight
                sum += dev_config.clutterDensity ;
                sum += dev_config.birthWeight ;

                // put normalizer in log form
                log_normalizer = safeLog(sum) ;
            }
            else
            {
                sum = dev_config.clutterDensity + dev_config.birthWeight ;
                log_normalizer = safeLog(sum) ;
            }

            // compute final feature weights
            for ( int j = 0 ; j < (n_features+1) ; j += blockDim.x )
            {
                int feature_idx = j + tid ;
                if ( feature_idx <= n_features)
                {
                    if ( feature_idx < n_features )
                    {
                        // update detection weight
                        updateIdx = feature_idx ;
                    }
                    else if ( feature_idx == n_features )
                    {
                        // update birth term weight
                        updateIdx = (n_measure-i)*n_features + i ;
//                            cuPrintf("%d\n",updateIdx) ;
                    }
                    ptr_update[updateIdx].weight =
                        exp(ptr_update[updateIdx].weight-log_normalizer) ;
                }
            }

            // update the pose particle weights
            if ( tid == 0 )
            {
//                    cuPrintf("normalizer = %f\n",log_normalizer) ;
                particle_weight += log_normalizer ;
            }
        }


        // Particle weighting
        __syncthreads() ;
        if ( tid == 0 )
        {
            if (dev_config.particleWeighting==0){
                particle_weight -= cardinality_predict ;
                particle_weights[map_idx] = particle_weight ;
            }
            else if (dev_config.particleWeighting==1){
                // vo empty map weighting
                // compute predicted cardinality
                float cn_predict = 0 ;
                for ( int i = 0 ; i < n_features ; i++ ){
                    cn_predict += features_predict[predict_offset+i].weight ;
                }

                // compute updated cardnality
                float cn_update = 0 ;
                for ( int i = 0 ; i < n_features*(n_measure+1) + n_measure ; i++){
                    cn_update += features_update[update_offset+i].weight ;
                }
                particle_weights[map_idx] = n_measure*dev_config.clutterDensity
                        + cn_update - cn_predict
                        - dev_config.clutterRate ;
            }
            else if (dev_config.particleWeighting == 2){
                // vo single feature weighting

                // find the feature with the highest single-object likelihood
                REAL max_weight = -FLT_MAX ;
                REAL max_j = -1 ;
                for ( int m = 0 ; m < n_measure ; m++){
                    for( int j = 0 ; j < n_features ; j++){
                        int preupdate_idx = preupdate_offset+m*preupdate_stride+j ;
                        if( j != max_j && features_preupdate[preupdate_idx].weight > max_weight){
                            max_weight = features_preupdate[preupdate_idx].weight ;
                            max_j = j ;
                        }
                    }
                }

//                Gaussian2D max_feature = features_predict[predict_offset+j] ;
//                // evaluate predicted PHD at maximum feature
//                REAL v_predict = 0 ;
//                for( int j = predict_offset ; j < predict_offset + n_features ; j++){
//                    v_predict += features_predict[j].weight +
//                            exp(max_feature.mean)
//                }
            }
        }

        // set the merging flags
        for ( int j = 0 ; j < n_update ; j+=blockDim.x)
        {
            int feature_idx = j+tid ;
            if (feature_idx < n_update)
            {
                int idx = update_offset+feature_idx ;
                if (features_update[idx].weight<dev_config.minFeatureWeight)
                    merge_flags[idx] = true ;
                else
                    merge_flags[idx] = false;
            }
        }
    }
}

__global__ void
phdUpdateKernelMixed(ConstantVelocityState* poses,
                     Gaussian2D* features_predict_static,
                     Gaussian4D* features_predict_dynamic,
                     int* map_offsets_static, int* map_offsets_dynamic,
                     int n_particles, int n_measure,
                     Gaussian2D* features_update_static,
                     Gaussian4D* features_update_dynamic,
                     bool* merge_flags_static, bool* merge_flags_dynamic,
                     REAL* particle_weights)
{
    __shared__ REAL sdata[256] ;

    int tid = threadIdx.x ;
    int feature_idx = 0 ;
    ConstantVelocityState pose ;
    int n_features_static = 0 ;
    int n_features_dynamic = 0 ;
    int predict_offset_static = 0 ;
    int predict_offset_dynamic = 0 ;
    int update_offset_static = 0 ;
    int update_offset_dynamic = 0 ;
    int n_update_static = 0 ;
    int n_update_dynamic = 0 ;
//    int preupdate_stride_static = map_offsets_static[n_particles] ;
//    int preupdate_stride_dynamic = map_offsets_dynamic[n_particles] ;

    REAL cardinality_predict = 0 ;
    REAL particle_weight = 0 ;

    // loop over particles
    for ( int map_idx = blockIdx.x ; map_idx < n_particles ; map_idx += gridDim.x )
    {
        if ( map_idx < n_particles )
        {
            // compute offsets for the current map
            n_features_static = map_offsets_static[map_idx+1]
                    - map_offsets_static[map_idx] ;
            n_features_dynamic = map_offsets_dynamic[map_idx+1]
                    - map_offsets_dynamic[map_idx] ;
            predict_offset_static = map_offsets_static[map_idx] ;
            predict_offset_dynamic = map_offsets_dynamic[map_idx] ;
            update_offset_static = predict_offset_static
                    + n_measure*predict_offset_static
                    + map_idx*n_measure ;
            update_offset_dynamic = predict_offset_dynamic
                    + n_measure*predict_offset_dynamic
                    + map_idx*n_measure ;
            n_update_static = n_features_static
                    + n_measure*n_features_static
                    + n_measure ;
            n_update_dynamic = n_features_dynamic
                    + n_measure*n_features_dynamic
                    + n_measure ;
            // get the corresponding vehicle pose
            pose = poses[map_idx] ;
            __syncthreads() ;

            // reinitialize predicted cardinality
            cardinality_predict = 0 ;

            // initialize log(particle_weight) update to 1
            particle_weight = 0 ;

            for ( int j = 0 ; j < (n_features_static+n_measure+n_features_dynamic+n_measure) ; j += blockDim.x )
            {
                feature_idx = j + tid ;

                // Distribution of features to threads:
                // [persistent_static | birth_static | persistent_dynamic | birth_dynamic ]
                REAL feature_pd = 0 ;
                REAL val = 0 ;
                bool is_static = (feature_idx < n_features_static+n_measure) ;
                bool is_dynamic = (feature_idx < n_features_static+n_measure+n_features_dynamic+n_measure)
                        && !is_static ;
                if ( is_static)
                {
                    Gaussian2D* ptr_update = NULL ;
                    Gaussian2D* ptr_nondetect = NULL ;
                    if(feature_idx < n_features_static)
                    {
                        ptr_nondetect = features_update_static
                                + update_offset_static + feature_idx ;
                        ptr_update = ptr_nondetect + n_features_static ;
                        computePreUpdate( pose, features_predict_static[predict_offset_static+feature_idx],
                                          n_features_static, n_measure, feature_pd,
                                          *ptr_nondetect, ptr_update ) ;
                        val = feature_pd
                                *features_predict_static[feature_idx].weight ;
                    }
                    else if (feature_idx < n_features_static+n_measure)
                    {
                        int z_idx = feature_idx - n_features_static ;
                        ptr_update = features_update_static + update_offset_static
                                + n_features_static
                                + n_measure*n_features_static + z_idx ;
                        computeBirth(pose, Z[z_idx],*ptr_update) ;
                    }
                }
                else if(is_dynamic)
                {
                    int feature_idx_dynamic = feature_idx
                            - n_features_static - n_measure ;
                    Gaussian4D* ptr_update = NULL ;
                    Gaussian4D* ptr_nondetect = NULL ;
                    if(feature_idx_dynamic < n_features_dynamic)
                    {
                        ptr_nondetect = features_update_dynamic
                                + update_offset_dynamic + feature_idx_dynamic ;
                        ptr_update = ptr_nondetect + n_features_dynamic ;
                        computePreUpdate( pose, features_predict_dynamic[predict_offset_dynamic+feature_idx_dynamic],
                                          n_features_dynamic, n_measure, feature_pd,
                                          *ptr_nondetect,ptr_update ) ;
                        val = feature_pd
                                *features_predict_dynamic[feature_idx_dynamic].weight ;
                    }
                    else if(feature_idx_dynamic < n_features_dynamic+n_measure )
                    {
                        int z_idx = feature_idx_dynamic - n_features_dynamic ;
                        ptr_update = features_update_dynamic + update_offset_dynamic
                                + n_features_dynamic
                                + n_features_dynamic*n_measure + z_idx ;
                        computeBirth(pose, Z[z_idx],*ptr_update) ;
//                        cuPrintf("Dynamic birth weight: %f\n",ptr_update->weight) ;
                        val = 0 ;
                    }
                }
                else
                {
                    // not a valid feature index
                    val = 0 ;
                }
                // compute predicted cardinality
                sumByReduction( sdata, val, tid );

                cardinality_predict += sdata[0] ;
                __syncthreads() ;
            }


            // finish updating weights - loop over measurements
            for ( int m = 0 ; m < n_measure ; m++ )
            {
                // pointers offset to updated features corresponding to current
                // measurement
                Gaussian2D* ptr_static = features_update_static
                        + update_offset_static
                        + n_features_static
                        + m*(n_features_static) ;
                Gaussian4D* ptr_dynamic = features_update_dynamic
                        + update_offset_dynamic
                        + n_features_dynamic
                        + m*(n_features_dynamic) ;
                REAL normalizer = 0 ;

                // normalizer is the sum of partially updated weights
                // corresponding to current measurement.
                for ( int j = 0 ; j < n_features_static+n_features_dynamic ; j += blockDim.x )
                {
                    feature_idx = j + tid ;
//                    REAL val = -FLT_MAX ;
                    REAL val = 0 ;

                    bool is_static = feature_idx < n_features_static ;
                    bool is_dynamic = (feature_idx < n_features_static+n_features_dynamic)
                            && !is_static ;

                    if ( is_static )
                        val = exp(ptr_static[feature_idx].weight) ;
                    else if(is_dynamic)
                        val = exp(ptr_dynamic[feature_idx-n_features_static].weight) ;

                    sumByReduction(sdata,val,tid);
                    normalizer += sdata[0] ;
                }
                normalizer += dev_config.clutterDensity
                        + dev_config.birthWeight ;

                // we get 2 birth terms when measurements are unlabeled
                if ( !dev_config.labeledMeasurements )
                    normalizer += dev_config.birthWeight ;

                normalizer = safeLog(normalizer) ;

                // loop through features corresponding to current measurement,
                // and divide by normalizer.
                for ( int j = 0 ; j < n_features_static+1+n_features_dynamic+1 ; j+=blockDim.x )
                {
                    feature_idx = j+tid ;
                    int idx_update = - 1 ;
                    bool is_static = (feature_idx < n_features_static+1) ;
                    bool is_dynamic = (feature_idx<(n_features_static+1+n_features_dynamic+1))
                            && ~is_static ;
                    if ( is_static)
                    {
                        int idx_update = -1 ;
                        if(feature_idx < n_features_static)
                        {
                            idx_update = feature_idx ;
                        }
                        else if (feature_idx == n_features_static)
                        {
                            idx_update = (n_measure-m)*n_features_static + m ;
                        }
                        ptr_static[idx_update].weight =
                                exp(ptr_static[idx_update].weight - normalizer) ;
                    }
                    else if(is_dynamic)
                    {
                        int feature_idx_dynamic = feature_idx
                                - n_features_static - 1 ;
                        if(feature_idx_dynamic < n_features_dynamic)
                        {
                            idx_update = feature_idx_dynamic ;
                        }
                        else if(feature_idx_dynamic==n_features_dynamic)
                        {
                            idx_update = (n_measure-m)*n_features_dynamic + m ;
                        }
                        ptr_dynamic[idx_update].weight =
                                exp(ptr_dynamic[idx_update].weight - normalizer) ;
                    }
                }

                // multiply particle weight update by normalizer
                __syncthreads() ;
                particle_weight += normalizer ;
            }

            // finish updating particle weight
            particle_weight -= cardinality_predict ;
            if ( tid == 0){
                if (dev_config.particleWeighting==0){
                    particle_weights[map_idx] = particle_weight ;
                }
                else if (dev_config.particleWeighting == 1){
                    // compute predicted cardinality
                    float cn_predict = 0 ;
                    for ( int i = 0 ; i < n_features_static ; i++ ){
                        cn_predict +=
                                features_predict_static[predict_offset_static+i].weight ;
                    }
                    for ( int i = 0 ; i < n_features_dynamic ; i++ ){
                        cn_predict +=
                                features_predict_dynamic[predict_offset_dynamic+i].weight ;
                    }
                    cn_predict += n_measure*dev_config.birthWeight ;

                    // compute updated cardnality
                    float cn_update = 0 ;
                    for ( int i = 0 ; i < n_features_static*(n_measure+1) + n_measure ; i++){
                        cn_update += features_update_static[update_offset_static+i].weight ;
                    }
                    for ( int i = 0 ; i < n_features_dynamic*(n_measure+1) + n_measure ; i++){
                        cn_update += features_update_dynamic[update_offset_dynamic+i].weight ;
                    }
                    particle_weights[map_idx] = n_measure*dev_config.clutterDensity
                            + cn_update - cn_predict
                            - dev_config.clutterRate ;
                }
                else if (dev_config.particleWeighting == 2){
//                    // vo single feature weighting

//                    // find the feature with the highest single-object likelihood
//                    REAL max_weight = -FLT_MAX ;
//                    REAL max_j = -1 ;
//                    for ( int m = 0 ; m < n_measure ; m++){
//                        for( int j = 0 ; j < n_features_static ; j++){
//                            int preupdate_idx = preupdate+m*preupdate_stride+j ;
//                            if( j != max_j && features_preupdate[preupdate_idx].weight > max_weight){
//                                max_weight = features_preupdate[preupdate_idx].weight ;
//                                max_j = j ;
//                            }
//                        }
//                    }

    //                Gaussian2D max_feature = features_predict[predict_offset+j] ;
    //                // evaluate predicted PHD at maximum feature
    //                REAL v_predict = 0 ;
    //                for( int j = predict_offset ; j < predict_offset + n_features ; j++){
    //                    v_predict += features_predict[j].weight +
    //                            exp(max_feature.mean)
    //                }

                }
            }
        }

        // set the merging flags
        for ( int j = 0 ; j < n_update_static+n_update_dynamic ; j+=blockDim.x)
        {
            int feature_idx = j+tid ;
            bool is_static = (feature_idx < n_update_static) ;
            bool is_dynamic = (feature_idx < n_update_static+n_update_dynamic)
                    && !is_static ;
            if (is_static)
            {
                if (features_update_static[update_offset_static+feature_idx].weight<dev_config.minFeatureWeight)
                    merge_flags_static[update_offset_static+feature_idx] = true ;
                else
                    merge_flags_static[update_offset_static+feature_idx] = false;
            }
            else if(is_dynamic)
            {
                feature_idx = feature_idx-n_update_static ;
                if (features_update_dynamic[update_offset_dynamic+feature_idx].weight<dev_config.minFeatureWeight)
                    merge_flags_dynamic[update_offset_dynamic+feature_idx] = true;
                else
                    merge_flags_dynamic[update_offset_dynamic+feature_idx] = false;
            }
        }
    }
}

/// Compute the integrated variance among the updated features.
/**
 * Non-detection, detection, and birth terms are sampled and summed.
 * Block operates on a single parent particle, and each thread
 * computes a single sample for each term.
 */
template <class GaussianType>
__global__ void
phdVarianceKernel(GaussianType* updated_features, REAL* variances,
                  int n_measure, int* map_offsets, int n_particles)
{
    // initialize local varaibles
    unsigned int predict_offset = 0 ;
    unsigned int update_offset = 0 ;
    unsigned int n_features = 0 ;
    unsigned int n_update = 0 ;

    // initialized shared memory
    __shared__ REAL sdata[256] ;

    unsigned int tid = threadIdx.x + blockDim.y*threadIdx.y ;

    for ( int map_idx = blockIdx.x ; map_idx < n_particles ; map_idx += gridDim.x )
    {
        // initialize map-specific variables
        predict_offset = map_offsets[map_idx] ;
        update_offset = predict_offset*(n_measure+1) +
                map_idx*n_measure ;
        n_features = map_offsets[map_idx+1] - map_offsets[map_idx] ;
        n_update = (n_features)*(n_measure+1) + n_measure ;

        // initialize output
        if (threadIdx.x == 0)
            variances[map_idx] = 0 ;

        REAL sum = 0 ;

        // make a local copy of the RNG state
        curandStateMRG32k3a_t rng_state = devRNGStates[threadIdx.x] ;
        skipahead((unsigned long long)tid,&rng_state);

        // sample and sum all update components
        for ( unsigned int i = 0 ; i < n_update ; i++){
            unsigned int idx = update_offset + i ;
            REAL val = sampleAndEvalGaussian(updated_features[idx],&rng_state) ;
            // non-detection term: just add the value
            if ( i < n_features ){
                sum += val ;
            }
            // detection or birth term, add x(1-x)
            else{
                sum += (1-val)*val ;
            }
        }

        // update the RNG state in global memory
        if (blockIdx.x == 0)
            devRNGStates[threadIdx.x] = rng_state ;

        // compute final sum
        for ( unsigned int i = 0 ; i < dev_config.nSamples ; i+=256){
            if (threadIdx.x >= i && threadIdx.x < i+256){
                sumByReduction(sdata,sum,tid-i);
            }
            if (threadIdx.x == 0)
                variances[map_idx] += sdata[0] ;
        }
    }
}

template <class GaussianType>
__global__ void
phdUpdateMergeKernel(GaussianType* updated_features,
                     GaussianType* mergedFeatures, int *mergedSizes,
                     bool *mergedFlags, int* map_offsets, int n_particles )
{
    __shared__ GaussianType maxFeature ;
    __shared__ GaussianType mergedFeature ;
    __shared__ REAL sdata[256] ;
    __shared__ int mergedSize ;
    __shared__ int update_offset ;
    __shared__ int n_update ;
    int tid = threadIdx.x ;
    REAL dist ;
    GaussianType feature ;
    clearGaussian(feature) ;
    int dims = getGaussianDim(feature) ;

    // loop over particles
    for ( int p = 0 ; p < n_particles ; p += gridDim.x )
    {
        int map_idx = p + blockIdx.x ;
        if ( map_idx < n_particles )
        {
            // initialize shared vars
            if ( tid == 0)
            {
                update_offset = map_offsets[map_idx] ;
                n_update = map_offsets[map_idx+1] - map_offsets[map_idx] ;
                mergedSize = 0 ;
            }
            __syncthreads() ;
            while(true)
            {
                // initialize the output values to defaults
                if ( tid == 0 )
                {
                    maxFeature.weight = -1 ;
                    clearGaussian(mergedFeature) ;
                }
                sdata[tid] = -1 ;
                __syncthreads() ;
                // find the maximum feature with parallel reduction
                for ( int i = update_offset ; i < update_offset + n_update ; i += blockDim.x)
                {
                    int idx = i + tid ;
                    if ( idx < (update_offset + n_update) )
                    {
                        if( !mergedFlags[idx] )
                        {
                            if (sdata[tid] == -1 ||
                                updated_features[(unsigned int)sdata[tid]].weight < updated_features[idx].weight )
                            {
                                sdata[tid] = idx ;
                            }
                        }
                    }
                }
                __syncthreads() ;
                for ( int s = blockDim.x/2 ; s > 0 ; s >>= 1 )
                {
                    if ( tid < s )
                    {
                        if ( sdata[tid] == -1 )
                            sdata[tid] = sdata[tid+s] ;
                        else if ( sdata[tid+s] >= 0 )
                        {
                            if(updated_features[(unsigned int)sdata[tid]].weight <
                            updated_features[(unsigned int)sdata[tid+s]].weight )
                            {
                                sdata[tid] = sdata[tid+s] ;
                            }
                        }

                    }
                    __syncthreads() ;
                }
                if ( sdata[0] == -1 || maxFeature.weight == 0 )
                    break ;
                else if(tid == 0)
                    maxFeature = updated_features[ (unsigned int)sdata[0] ] ;
                __syncthreads() ;

                // find features to merge with max feature
                REAL sval0 = 0 ;
//                REAL sval1 = 0 ;
//                REAL sval2 = 0 ;
                clearGaussian(feature) ;
                for ( int i = update_offset ; i < update_offset + n_update ; i += blockDim.x )
                {
                    int idx = tid + i ;
                    if ( idx < update_offset+n_update )
                    {
                        if ( !mergedFlags[idx] )
                        {
                            if ( dev_config.distanceMetric == 0 )
                                dist = computeMahalDist(maxFeature, updated_features[idx]) ;
                            else if ( dev_config.distanceMetric == 1)
                                dist = computeHellingerDist(maxFeature, updated_features[idx]) ;
                            if ( dist < dev_config.minSeparation )
                            {
                                feature.weight += updated_features[idx].weight ;
                                for ( int j = 0 ; j < dims ; j++ )
                                    feature.mean[j] += updated_features[idx].weight*updated_features[idx].mean[j] ;
                            }
                        }
                    }
                }
                // merge means and weights
                sval0 = feature.weight ;
                sumByReduction(sdata, sval0, tid) ;
                if ( tid == 0 )
                    mergedFeature.weight = sdata[0] ;
                __syncthreads() ;
                if ( mergedFeature.weight == 0 )
                    break ;
                for ( int j = 0 ; j < dims ; j++ )
                {
                    sval0 = feature.mean[j] ;
                    sumByReduction(sdata,sval0,tid);
                    if( tid == 0 )
                        mergedFeature.mean[j] = sdata[0]/mergedFeature.weight ;
                    __syncthreads() ;
                }

                // merge the covariances
                sval0 = 0 ;
//                sval1 = 0 ;
//                sval2 = 0 ;
                clearGaussian(feature) ;
                for ( int i = update_offset ; i < update_offset+n_update ; i += blockDim.x )
                {
                    int idx = tid + i ;
                    if ( idx < update_offset+n_update )
                    {
                        if (!mergedFlags[idx])
                        {
                            if ( dev_config.distanceMetric == 0 )
                                dist = computeMahalDist(maxFeature, updated_features[idx]) ;
                            else if ( dev_config.distanceMetric == 1)
                                dist = computeHellingerDist(maxFeature, updated_features[idx]) ;
                            if ( dist < dev_config.minSeparation )
                            {
                                // use the mean of the local gaussian variable
                                // to store the innovation vector
                                for (int j = 0 ; j < dims ; j++)
                                {
                                    feature.mean[j] = mergedFeature.mean[j]
                                            - updated_features[idx].mean[j] ;
                                }
                                for (int j = 0 ; j < dims ; j++ )
                                {
                                    REAL outer = feature.mean[j] ;
                                    for ( int k = 0 ; k < dims ; k++)
                                    {
                                        REAL inner = feature.mean[k] ;
                                        feature.cov[j*dims+k] +=
                                                updated_features[idx].weight*
                                                (updated_features[idx].cov[j*dims+k]
                                                + outer*inner) ;
                                    }
                                }
                                mergedFlags[idx] = true ;
                            }
                        }
                    }
                }
                for ( int j = 0 ; j < dims*dims ; j++)
                {
                    sval0 = feature.cov[j] ;
                    sumByReduction(sdata,sval0,tid);
                    if ( tid == 0 )
                        mergedFeature.cov[j] = sdata[0]/mergedFeature.weight ;
                    __syncthreads() ;
                }
                if ( tid == 0 )
                {
                    force_symmetric_covariance(mergedFeature) ;
                    int mergeIdx = update_offset + mergedSize ;
                    copy_gaussians(mergedFeature,mergedFeatures[mergeIdx]) ;
                    mergedSize++ ;
                }
                __syncthreads() ;
            }
            __syncthreads() ;
            // save the merged map size
            if ( tid == 0 )
                mergedSizes[map_idx] = mergedSize ;
        }
    } // end loop over particles
    return ;
}


template <class GaussianType>
void
prepareUpdateInputs(vector<vector<GaussianType> > maps,
                    ConstantVelocityState* dev_poses,
                    int n_particles, int n_measure,
                    GaussianType*& dev_maps_inrange,
                    int*& dev_map_offsets, GaussianType*& dev_maps_updated,
                    bool*& dev_merged_flags,
                    vector<GaussianType>& features_in,
                    vector<GaussianType>& features_out1 ,
                    vector<GaussianType>& features_out2,
                    vector<int>& n_in_range_vec,
                    vector<int>& n_out_range1_vec,
                    vector<int>& n_out_range2_vec )
{
    //------- Variable Declarations ---------//

    vector<GaussianType> concat ;
    vector<int> map_sizes(n_particles) ;
    int nThreads = 0 ;

    // map offsets
    vector<int> map_offsets_in(n_particles+1,0) ;
    vector<int> map_offsets_out(n_particles+1,0) ;

    // device variables
    GaussianType* dev_maps = NULL ;
    int* dev_map_sizes = NULL ;
    int* dev_n_in_range = NULL ;
    int* dev_n_out_range2 = NULL ;
    char* dev_in_range = NULL ;
    int total_features = 0 ;

    // in/out range book-keeping variables
    int n_in_range = 0 ;
    int n_out_range = 0 ;
    int idx_in = 0 ;
    int idx_out = 0 ;
    int idx_out2 = 0 ;
    int n_out_range1 = 0 ;
    int n_out_range2 = 0 ;
    vector<char> in_range ;

    //------- End Variable Declarations -----//


    ///////////////////////////////////////////////////////////////////////////
    //
    // concatenate all the maps together for parallel processing
    //
    ///////////////////////////////////////////////////////////////////////////
    for ( unsigned int n = 0 ; n < n_particles ; n++ )
    {
        concat.insert( concat.end(),
                        maps[n].begin(),
                        maps[n].end() ) ;
        map_sizes[n] = maps[n].size() ;

        // keep track of largest map feature count
        if ( map_sizes[n] > nThreads )
            nThreads = map_sizes[n] ;
        nThreads = min(nThreads,256) ;
        total_features += map_sizes[n] ;
    }

    // allocate device space for map sizes
    checkCudaErrors(
                cudaMalloc( (void**)&dev_map_sizes,
                            n_particles*sizeof(int) ) ) ;

    if ( total_features > 0)
    {
        ///////////////////////////////////////////////////////////////////////
        //
        // split features into in/out range parts
        //
        ///////////////////////////////////////////////////////////////////////

        // allocate device memory
        checkCudaErrors(
                    cudaMalloc( (void**)&dev_maps,
                                total_features*sizeof(GaussianType) ) ) ;;
        checkCudaErrors(
                    cudaMalloc( (void**)&dev_n_in_range,
                                n_particles*sizeof(int) ) ) ;
        checkCudaErrors(
                    cudaMalloc( (void**)&dev_n_out_range2,
                                n_particles*sizeof(int) ) ) ;
        checkCudaErrors(
                    cudaMalloc( (void**)&dev_in_range,
                                total_features*sizeof(char) ) ) ;


        // copy inputs
        checkCudaErrors(
            cudaMemcpy( dev_maps, &concat[0], total_features*sizeof(GaussianType),
                        cudaMemcpyHostToDevice )
        ) ;
        checkCudaErrors(
                    cudaMemcpy( dev_map_sizes, &map_sizes[0], n_particles*sizeof(int),
                        cudaMemcpyHostToDevice )
        ) ;


        // kernel launch
        DEBUG_MSG("launching computeInRangeKernel") ;
        DEBUG_VAL(nThreads) ;
        computeInRangeKernel<<<n_particles,nThreads>>>
            ( dev_maps, dev_map_sizes, n_particles, dev_poses, dev_in_range,
              dev_n_in_range, dev_n_out_range2 ) ;

        // allocate outputs
        in_range.resize(total_features);

        // copy outputs
        checkCudaErrors(
                    cudaMemcpy( &in_range[0],dev_in_range,
                                total_features*sizeof(char),
                                cudaMemcpyDeviceToHost )
        ) ;
        checkCudaErrors(
            cudaMemcpy( &n_in_range_vec[0],dev_n_in_range,n_particles*sizeof(int),
                        cudaMemcpyDeviceToHost )
        ) ;
        checkCudaErrors(
            cudaMemcpy( &n_out_range2_vec[0],dev_n_out_range2,n_particles*sizeof(int),
                        cudaMemcpyDeviceToHost )
        ) ;

        // get total number of in-range features
        for ( int i = 0 ; i < n_particles ; i++ )
        {
            n_in_range += n_in_range_vec[i] ;
            n_out_range1_vec[i] = maps[i].size() -  n_in_range_vec[i]
                    - n_out_range2_vec[i] ;
            n_out_range2 += n_out_range2_vec[i] ;
        }

        // divide features into in-range/out-of-range parts
        n_out_range = total_features - n_in_range ;
        n_out_range1 = n_out_range - n_out_range2 ;
        DEBUG_VAL(n_in_range) ;
        DEBUG_VAL(n_out_range1) ;
        DEBUG_VAL(n_out_range2) ;
        features_in.resize(n_in_range) ;
        features_out1.resize(n_out_range1) ;
        features_out2.resize(n_out_range2) ;
        for ( int i = 0 ; i < total_features ; i++ )
        {
            if (in_range[i] == 1)
                features_in[idx_in++] = concat[i] ;
            else if (in_range[i] == 2 )
                features_out2[idx_out2++] = concat[i] ;
            else
                features_out1[idx_out++] = concat[i] ;
        }

        // free memory
        checkCudaErrors( cudaFree( dev_maps ) ) ;
        checkCudaErrors( cudaFree( dev_in_range ) ) ;
        checkCudaErrors( cudaFree( dev_n_in_range ) ) ;


        // perform an (inclusive) prefix scan on the map sizes to determine indexing
        // offsets for each map
        for ( int i = 1 ; i < n_particles+1 ; i++ )
        {
            map_offsets_in[i] = map_offsets_in[i-1] + n_in_range_vec[i-1] ;
            map_offsets_out[i] = map_offsets_out[i-1] + n_in_range_vec[i-1] ;
        }
    }

    /************************************************
     *
     *  Prepare PHD update inputs
     *
     ************************************************/
    int n_update = n_in_range*(n_measure+1) + n_measure*n_particles ;

    // allocate device memory
    checkCudaErrors(
                    cudaMalloc( (void**)&dev_maps_inrange,
                                n_in_range*sizeof(GaussianType) ) ) ;
    checkCudaErrors(
                    cudaMalloc( (void**)&dev_map_offsets,
                                (n_particles+1)*sizeof(int) ) ) ;
    checkCudaErrors(
                cudaMalloc((void**)&dev_maps_updated,
                           n_update*sizeof(GaussianType)) ) ;
    checkCudaErrors(
                cudaMalloc((void**)&dev_merged_flags,
                           n_update*sizeof(bool)) ) ;

    // copy inputs
    checkCudaErrors(
        cudaMemcpy( dev_maps_inrange, &features_in[0],
                    n_in_range*sizeof(GaussianType),
                    cudaMemcpyHostToDevice )
    ) ;
    checkCudaErrors( cudaMemcpy( dev_map_offsets, &map_offsets_in[0],
                                (n_particles+1)*sizeof(int),
                                cudaMemcpyHostToDevice ) ) ;
}

template <class GaussianType>
/**
 * @brief pruneMap Prune a gaussian mixture.
 *
 * The elements of dev_maps whose corresponding flag equals true are removed
 * and the resulting array is written back into dev_maps. dev_merged_flags is
 * also overwritten with an array of the appropriate number of false elements.
 * map_sizes is overwritten with the sizes of the pruned maps
 *
 * @param dev_maps Device pointer to array of gaussian features
 * @param dev_merged_flags Device array of boolean flags, true = prune.
 * @param map_sizes Vector of map sizes
 * @param n_gaussians Total number of gaussians.
 * @return Total number of gaussians after pruning
 */
int
pruneMap(GaussianType*& dev_maps,
         bool*& dev_merged_flags,
         std::vector<int>& map_sizes,
         int n_gaussians){

    // wrap pointers in thrust types
    thrust::device_ptr<GaussianType> ptr_maps(dev_maps) ;
    thrust::device_ptr<bool> ptr_flags(dev_merged_flags) ;

    // create the output vector, with same size as the input
    thrust::device_vector<GaussianType> dev_pruned(n_gaussians) ;

    // do the pruning
    thrust::remove_copy_if(ptr_maps,ptr_maps+n_gaussians,
                    ptr_flags,
                    dev_pruned.begin(),
                    thrust::identity<bool>()) ;

    // recalculate map sizes
    int n_particles = map_sizes.size() ;
    std::vector<int> map_sizes_pruned(n_particles,0) ;
    host_vector<bool> flags(ptr_flags,ptr_flags+n_gaussians) ;
    int n = 0 ;
    int n_pruned = 0 ;
    for ( int i = 0 ; i < n_particles ; i++){
        for( int j = 0 ; j < map_sizes[i] ; j++){
            if (!flags[n++]){
                map_sizes_pruned[i]++ ;
                n_pruned++ ;
            }
        }
    }

//    cout << "pruned features: " << endl ;
//    for ( int i = 0 ; i < n_pruned ; i++ ){
//        GaussianType g = dev_pruned[i] ;
//        print_feature(g) ;
//    }

    // store pruned results
    thrust::device_free(ptr_maps) ;
    ptr_maps = thrust::device_malloc<GaussianType>(n_pruned) ;
    thrust::copy_n(dev_pruned.begin(),n_pruned,ptr_maps) ;
    dev_maps = raw_pointer_cast(ptr_maps) ;

    thrust::device_free(ptr_flags) ;
    ptr_flags = thrust::device_malloc<bool>(n_pruned) ;
    thrust::fill(ptr_flags,ptr_flags+n_pruned,false) ;
    dev_merged_flags = raw_pointer_cast(ptr_flags) ;

    map_sizes = map_sizes_pruned ;

    return n_pruned ;
}

template <class GaussianType>
void
mergeAndCopyMaps(GaussianType*& dev_maps_updated,
                 bool*& dev_merged_flags,
                 vector<GaussianType> features_out1,
                 vector<GaussianType> features_out2,
                 vector<int> n_in_range_vec,
                 vector<int> n_out_range1_vec,
                 vector<int> n_out_range2_vec,
                 int n_particles, int n_measure, int n_update,
                 vector<vector<GaussianType> >& maps_output )
{
    vector<int> map_offsets(n_particles+1) ;
    size_t combined_size ;
    GaussianType* maps_merged = NULL ;
    int* map_sizes_merged = NULL ;

    int offset = 0 ;
    int offset_updated = 0 ;
    int offset_out = 0 ;

    // device variables
    GaussianType* dev_maps_merged = NULL ;
    GaussianType* dev_maps_combined = NULL ;
    bool* dev_merged_flags_combined = NULL ;
    int* dev_n_merged = NULL ;
    int* dev_map_offsets = NULL ;

    int n_out_range1 = features_out1.size() ;
    int n_out_range2 = features_out2.size() ;

    // prune low-weighted features
    DEBUG_VAL(n_update) ;
    vector<int> map_sizes_inrange(n_particles) ;
    for ( int n = 0 ; n < n_particles ; n++){
        map_sizes_inrange[n] = n_in_range_vec[n]*(n_measure+1) + n_measure ;
    }
    int n_pruned = pruneMap(dev_maps_updated,dev_merged_flags,
                            map_sizes_inrange,n_update) ;
    DEBUG_VAL(n_pruned) ;


    // recombine updated in-range map with nearly in-range map do merging
    DEBUG_MSG("Recombining maps") ;
    combined_size = (n_pruned+n_out_range2)*sizeof(GaussianType) ;
    checkCudaErrors( cudaMalloc( (void**)&dev_maps_combined, combined_size ) ) ;
    checkCudaErrors( cudaMalloc( (void**)&dev_merged_flags_combined,
                                (n_pruned+n_out_range2)*sizeof(bool) ) ) ;


    map_offsets[0] = 0 ;
    for ( int n = 0 ; n < n_particles ; n++ )
    {
        // in-range map for particle n
        int n_in_range_n = map_sizes_inrange[n] ;
        checkCudaErrors( cudaMemcpy( dev_maps_combined+offset,
                                    dev_maps_updated+offset_updated,
                                    n_in_range_n*sizeof(GaussianType),
                                    cudaMemcpyDeviceToDevice) ) ;
        checkCudaErrors( cudaMemcpy( dev_merged_flags_combined+offset,
                                    dev_merged_flags+offset_updated,
                                    n_in_range_n*sizeof(bool)
                                    ,cudaMemcpyDeviceToDevice ) ) ;
        offset += n_in_range_n ;
        offset_updated += n_in_range_n ;

        // nearly in range map for particle n
        vector<char> merged_flags_out(n_out_range2_vec[n],0) ;
        checkCudaErrors( cudaMemcpy( dev_maps_combined+offset,
                                    &features_out2[offset_out],
                                    n_out_range2_vec[n]*sizeof(GaussianType),
                                    cudaMemcpyHostToDevice ) ) ;
        checkCudaErrors( cudaMemcpy( dev_merged_flags_combined+offset,
                                    &merged_flags_out[0],
                                    n_out_range2_vec[n]*sizeof(bool),
                                    cudaMemcpyHostToDevice) ) ;
        offset += n_out_range2_vec[n] ;
        offset_out += n_out_range2_vec[n] ;


        map_offsets[n+1] = offset ;
    }

    DEBUG_VAL(combined_size) ;
    checkCudaErrors( cudaMalloc((void**)&dev_maps_merged,
                           combined_size ) ) ;
    checkCudaErrors( cudaMalloc((void**)&dev_n_merged,
                               n_particles*sizeof(int) ) ) ;
    checkCudaErrors( cudaMalloc((void**)&dev_map_offsets,
                               (n_particles+1)*sizeof(int) ) ) ;
    checkCudaErrors( cudaMemcpy( dev_map_offsets, &map_offsets[0],
                                (n_particles+1)*sizeof(int),
                                cudaMemcpyHostToDevice ) ) ;
    //


    thrust::device_ptr<bool> ptr_flags(dev_merged_flags_combined) ;
    thrust::fill(ptr_flags, ptr_flags+n_pruned+n_out_range2,false) ;

    DEBUG_MSG("launching phdUpdateMergeKernel") ;
    phdUpdateMergeKernel<<<n_particles,256>>>
        ( dev_maps_combined, dev_maps_merged, dev_n_merged,
          dev_merged_flags_combined, dev_map_offsets, n_particles ) ;
    checkCudaErrors( cudaDeviceSynchronize() ) ;

//    // copy one feature and look at it
//    GaussianType feature_test ;
//    checkCudaErrors(cudaMemcpy(&feature_test,dev_maps_merged,sizeof(GaussianType),cudaMemcpyDeviceToHost) ) ;
//    cout << "first merged feature: " << endl ;
//    print_feature(feature_test) ;

    // allocate outputs
    DEBUG_MSG("Allocating update and merge outputs") ;
    maps_merged = (GaussianType*)malloc( combined_size ) ;
    map_sizes_merged = (int*)malloc( n_particles*sizeof(int) ) ;

    // copy outputs
    checkCudaErrors(
                cudaMemcpy( maps_merged, dev_maps_merged,
                            combined_size,
                            cudaMemcpyDeviceToHost ) ) ;
    checkCudaErrors(
                cudaMemcpy( map_sizes_merged, dev_n_merged,
                            n_particles*sizeof(int),
                            cudaMemcpyDeviceToHost ) ) ;

    offset_updated = 0 ;
    offset_out = 0 ;
    for ( int i = 0 ; i < n_particles ; i++ )
    {
        offset_updated = map_offsets[i] ;
//        DEBUG_VAL(map_sizes_merged[i]) ;
        maps_output[i].assign(maps_merged+offset_updated,
                            maps_merged+offset_updated+map_sizes_merged[i]) ;

        // recombine with out-of-range features, if any
        if ( n_out_range1 > 0 && n_out_range1_vec[i] > 0 )
        {
            maps_output[i].insert( maps_output[i].end(),
                                    features_out1.begin()+offset_out,
                                    features_out1.begin()+offset_out+n_out_range1_vec[i] ) ;
            offset_out += n_out_range1_vec[i] ;
        }
//        cout << "Merged map " << i << endl ;
//        for ( int j = 0 ; j < maps_output[i].size() ; j++ ){
//            print_feature(maps_output[i][j]) ;
//        }
    }

    free(maps_merged) ;
    free(map_sizes_merged) ;
    checkCudaErrors( cudaFree( dev_maps_combined ) ) ;
    checkCudaErrors( cudaFree( dev_maps_merged ) ) ;
    checkCudaErrors( cudaFree( dev_merged_flags_combined ) ) ;
    checkCudaErrors( cudaFree( dev_n_merged ) ) ;
    checkCudaErrors( cudaFree( dev_maps_updated) ) ;
    checkCudaErrors( cudaFree( dev_merged_flags) ) ;
}


SynthSLAM
phdUpdateSynth(SynthSLAM& particles, measurementSet measurements)
{
    //------- Variable Declarations ---------//

    int n_measure = 0 ;
    int n_particles = particles.n_particles ;
    DEBUG_VAL(n_particles) ;
    vector<int> map_sizes_static(n_particles,0) ;
    vector<int> map_sizes_dynamic(n_particles,0) ;    

    // map offsets
    vector<int> map_offsets_in_static(n_particles+1,0) ;
    vector<int> map_offsets_out_static(n_particles+1,0) ;

    SynthSLAM particlesPreMerge(particles) ;

    // device variables
    ConstantVelocityState* dev_poses = NULL ;
    int *dev_map_offsets_static = NULL ;
    int *dev_map_offsets_dynamic = NULL ;
    Gaussian2D* dev_maps_inrange_static = NULL ;
    Gaussian4D* dev_maps_inrange_dynamic = NULL ;
    Gaussian2D* dev_maps_updated_static = NULL ;
    Gaussian4D* dev_maps_updated_dynamic = NULL ;
    REAL* dev_particle_weights = NULL ;
    bool* dev_merged_flags_static = NULL ;
    bool* dev_merged_flags_dynamic = NULL ;

    // in/out range book-keeping variables
    vector<char> in_range ;
    vector<int> n_in_range_vec_static(n_particles,0) ;
    vector<int> n_in_range_vec_dynamic(n_particles,0) ;
    vector<int> n_out_range1_vec_static(n_particles,0) ;
    vector<int> n_out_range1_vec_dynamic(n_particles,0) ;
    vector<int> n_out_range2_vec_static(n_particles,0) ;
    vector<int> n_out_range2_vec_dynamic(n_particles,0) ;
    vector<Gaussian2D> features_in_static ;
    vector<Gaussian2D> features_out1_static ;
    vector<Gaussian2D> features_out2_static ;
    vector<Gaussian4D> features_in_dynamic ;
    vector<Gaussian4D> features_out1_dynamic ;
    vector<Gaussian4D> features_out2_dynamic ;


    // output variables

    //------- End Variable Declarations -----//

    // make a copy of the particles
    particlesPreMerge = particles ;

    // check for memory limit for storing measurements in constant mem
    n_measure = measurements.size() ;
    if ( n_measure > 256 )
    {
        DEBUG_MSG("Warning: maximum number of measurements per time step exceeded") ;
        n_measure = 256 ;
    }
    DEBUG_VAL(n_measure) ;

    // copy measurements to device
    checkCudaErrors(
                cudaMemcpyToSymbol( Z, &measurements[0],
                                    n_measure*sizeof(RangeBearingMeasurement) ) ) ;

    // copy particle poses to device
    checkCudaErrors(
            cudaMalloc( (void**)&dev_poses,
                        n_particles*sizeof(ConstantVelocityState) ) ) ;
    checkCudaErrors(
                cudaMemcpy(dev_poses,&particles.states[0],
                           n_particles*sizeof(ConstantVelocityState),
                           cudaMemcpyHostToDevice) ) ;

    // extract in-range portions of maps, and allocate output arrays
    if(config.featureModel==STATIC_MODEL
            || config.featureModel==MIXED_MODEL)
    {
        prepareUpdateInputs( particles.maps_static,
                             dev_poses, n_particles, n_measure,
                             dev_maps_inrange_static, dev_map_offsets_static,
                             dev_maps_updated_static, dev_merged_flags_static,
                             features_in_static, features_out1_static,
                             features_out2_static, n_in_range_vec_static,
                             n_out_range1_vec_static, n_out_range2_vec_static) ;
    }
    if(config.featureModel == DYNAMIC_MODEL
            || config.featureModel == MIXED_MODEL)
    {
        prepareUpdateInputs( particles.maps_dynamic,
                             dev_poses, n_particles, n_measure,
                             dev_maps_inrange_dynamic, dev_map_offsets_dynamic,
                             dev_maps_updated_dynamic, dev_merged_flags_dynamic,
                             features_in_dynamic, features_out1_dynamic,
                             features_out2_dynamic, n_in_range_vec_dynamic,
                             n_out_range1_vec_dynamic,n_out_range2_vec_dynamic) ;
    }

    // allocate arrays for particle weight update
    checkCudaErrors(
                cudaMalloc((void**)&dev_particle_weights,
                           n_particles*sizeof(REAL) ) ) ;


    // launch kernel
    int nBlocks = min(n_particles,32768) ;
    int n_update_static = features_in_static.size()*(n_measure+1)
            + n_measure*n_particles ;
    int n_update_dynamic = features_in_dynamic.size()*(n_measure+1)
            + n_measure*n_particles ;

    cudaPrintfInit(4194304) ;
    if(config.featureModel == MIXED_MODEL)
    {
        DEBUG_MSG("launching phdUpdateKernelMixed") ;
        phdUpdateKernelMixed<<<nBlocks,256>>>(
            dev_poses, dev_maps_inrange_static, dev_maps_inrange_dynamic,
            dev_map_offsets_static, dev_map_offsets_dynamic,
            n_particles,n_measure,
            dev_maps_updated_static, dev_maps_updated_dynamic,
            dev_merged_flags_static, dev_merged_flags_dynamic,
            dev_particle_weights);
        checkCudaErrors( cudaDeviceSynchronize() ) ;
        checkCudaErrors( cudaFree( dev_maps_inrange_dynamic ) ) ;
        checkCudaErrors( cudaFree( dev_map_offsets_dynamic ) ) ;
    }
    else if(config.featureModel==STATIC_MODEL)
    {
        DEBUG_MSG("Computing Birth terms") ;
        int n_births = n_particles*n_measure ;
        vector<Gaussian2D> births(n_births) ;
        for ( int i = 0 ; i < n_particles ; i++){
            ConstantVelocityState pose = particles.states[i] ;
            for( int j = 0 ; j < n_measure ; j++){
                int idx = i*n_measure + j ;
                RangeBearingMeasurement z = measurements[j] ;

                // invert measurement
                REAL theta = pose.ptheta + z.bearing ;
                REAL dx = z.range*cos(theta) ;
                REAL dy = z.range*sin(theta) ;
                births[idx].mean[0] = pose.px + dx ;
                births[idx].mean[1] = pose.py + dy ;

                // inverse measurement jacobian
                REAL J[4] ;
                J[0] = dx/z.range ;
                J[1] = dy/z.range ;
                J[2] = -dy ;
                J[3] = dx ;

                // measurement noise
                REAL var_range = pow(config.stdRange*config.birthNoiseFactor,2) ;
                REAL var_bearing = pow(config.stdBearing*config.birthNoiseFactor,2) ;

                // compute birth covariance
                births[idx].cov[0] = pow(J[0],2)*var_range +
                        pow(J[2],2)*var_bearing ;
                births[idx].cov[1] = J[0]*J[1]*var_range +
                        J[2]*J[3]*var_bearing ;
                births[idx].cov[2] =
                        births[idx].cov[1] ;
                births[idx].cov[3] = pow(J[1],2)*var_range +
                        pow(J[3],2)*var_bearing ;

                // set birth weight
                if(z.label==STATIC_MEASUREMENT || !config.labeledMeasurements)
                    births[idx].weight = safeLog(config.birthWeight) ;
                else
                    births[idx].weight = safeLog(0) ;

//                print_feature(births[idx]) ;
            }
        }
        Gaussian2D* dev_births = NULL ;
        checkCudaErrors(cudaMalloc(
                           (void**)&dev_births,
                           n_births*sizeof(Gaussian2D))) ;
        checkCudaErrors(cudaMemcpy(
                           dev_births,&births[0],
                           n_births*sizeof(Gaussian2D),
                           cudaMemcpyHostToDevice)) ;

        DEBUG_MSG("Computing PHD preupdate") ;
        // allocate device memory for pre-updated features
        int n_features_total = features_in_static.size() ;
        int n_preupdate = n_features_total*n_measure ;
        DEBUG_VAL(n_preupdate) ;
        Gaussian2D* dev_features_preupdate = NULL ;
        checkCudaErrors(cudaMalloc((void**)&dev_features_preupdate,
                                  n_preupdate*sizeof(Gaussian2D))) ;


        // create pose index vector
        vector<int> pose_idx ;
        for (int i = 0 ; i < n_particles ; i++){
            pose_idx.insert(pose_idx.end(),n_in_range_vec_static[i],i) ;
        }
//        for ( int i = 0 ; i < pose_idx.size() ; i++){
//            DEBUG_VAL(pose_idx[i]) ;
//        }
        int* dev_pose_idx = NULL ;
        checkCudaErrors(cudaMalloc((void**)&dev_pose_idx,
                                  n_features_total*sizeof(int))) ;
        checkCudaErrors(cudaMemcpy(dev_pose_idx,&pose_idx[0],
                                  n_features_total*sizeof(int),
                                  cudaMemcpyHostToDevice)) ;

        // create pd vector
        REAL* dev_features_pd = NULL ;
        checkCudaErrors(cudaMalloc((void**)&dev_features_pd,
                                 n_features_total*sizeof(REAL))) ;
        // likelihoods array
        REAL* dev_likelihoods = NULL ;
        checkCudaErrors(cudaMalloc((void**)&dev_likelihoods,
                                  n_preupdate*sizeof(REAL) ) ) ;

        // call the preupdate kernel
        nBlocks = min(int(ceil(n_features_total/256.0)),65535) ;
        DEBUG_VAL(nBlocks) ;
        preUpdateSynthKernel<<<nBlocks,256>>>(
           dev_poses,dev_pose_idx,dev_maps_inrange_static,
           dev_features_pd,n_features_total,n_measure,
           dev_likelihoods,dev_features_preupdate) ;
        checkCudaErrors( cudaDeviceSynchronize() ) ;

//        // check preupdate terms
//        thrust::device_ptr<Gaussian2D> ptr_preupdate(dev_features_preupdate) ;
//        thrust::device_vector<Gaussian2D> dev_preupdate(ptr_preupdate,ptr_preupdate+n_preupdate) ;
//        thrust::host_vector<Gaussian2D> preupdate(dev_preupdate) ;
//        for ( int i = 0 ; i < preupdate.size() ; i++){
//            Gaussian2D g = preupdate[i] ;
//            print_feature(g) ;
//        }

        DEBUG_MSG("launching phdUpdateKernel Static") ;
        nBlocks = min(n_particles,65535) ;
        phdUpdateKernel<<<nBlocks,256>>>(
            dev_maps_inrange_static, dev_features_pd, dev_features_preupdate,
            dev_births, dev_map_offsets_static,n_particles,n_measure,
            dev_maps_updated_static,dev_merged_flags_static,
            dev_particle_weights ) ;
        checkCudaErrors( cudaDeviceSynchronize() ) ;
        cudaFree(dev_births) ;
        cudaFree(dev_pose_idx) ;
        cudaFree(dev_features_preupdate) ;
        cudaFree(dev_features_pd) ;

        // compute variance
        REAL* dev_variance = NULL ;
        checkCudaErrors( cudaMalloc((void**)&dev_variance,
                                    n_particles*sizeof(REAL) ) ) ;
        int n_threads = config.nSamples ;
        nBlocks = min(n_particles,65535) ;
        phdVarianceKernel<<<nBlocks,n_threads>>>(
            dev_maps_updated_static,dev_variance, n_measure,
            dev_map_offsets_static,n_particles) ;
        checkCudaErrors( cudaDeviceSynchronize() ) ;
        checkCudaErrors( cudaMemcpy(&particles.variances[0],dev_variance,
                n_particles*sizeof(REAL),cudaMemcpyDeviceToHost ) ) ;
        checkCudaErrors( cudaFree(dev_variance) ) ;

        // RBPHD single-feature likelihood
        if (config.particleWeighting == 2 && n_preupdate > 0){
            vector<REAL> likelihoods(n_preupdate) ;
            checkCudaErrors(cudaMemcpy(&likelihoods[0],dev_likelihoods,
                                      n_preupdate*sizeof(REAL),
                                      cudaMemcpyDeviceToHost)) ;

            checkCudaErrors(cudaMemcpy(&map_offsets_in_static[0],
                                      dev_map_offsets_static,
                                      (n_particles+1)*sizeof(int),
                                      cudaMemcpyDeviceToHost)) ;

            std::vector<Gaussian2D> max_features(n_particles) ;
            std::vector<REAL> predict_vals(n_particles) ;
            std::vector<REAL> max_likelihoods(n_particles) ;
            std::vector<REAL> cn_predict(n_particles) ;
            for ( int n = 0 ; n < n_particles ; n++){
                REAL max_likelihood = -FLT_MAX ;
                int max_i = 0 ;
                int max_m = 0 ;
                for ( int m = 0 ; m < n_measure ; m++){
                    int start = m*n_features_total + map_offsets_in_static[n] ;
                    int end = m*n_features_total + map_offsets_in_static[n+1] ;
                    for ( int i = start ; i < end ; i++ ){
                        if (likelihoods[i] > max_likelihood){
                            max_i = end - i ;
                            max_m = m ;
                            max_likelihood = likelihoods[i] ;
                        }
                    }
                    int n_features_n = map_offsets_in_static[n+1]
                            - map_offsets_in_static[n] ;
                    int offset_update = map_offsets_in_static[n]*(n_measure)
                            + n*n_measure ;
                    int idx_max_feature = offset_update + (max_m+1)*n_features_n
                            + max_i ;
                    checkCudaErrors(cudaMemcpy(&max_features[n],
                                              dev_maps_updated_static+idx_max_feature,
                                              sizeof(Gaussian2D),
                                              cudaMemcpyDeviceToHost)) ;
                }
                cn_predict[n] = sumGaussianMixtureWeights(particles.maps_static[n]) ;
                max_likelihoods[n] = max_likelihood ;
                predict_vals[n] = evalGaussianMixture(particles.maps_static[n],
                                                       max_features[n].mean) ;
            }
            mergeAndCopyMaps( dev_maps_updated_static,dev_merged_flags_static,
                              features_out1_static,
                              features_out2_static, n_in_range_vec_static,
                              n_out_range1_vec_static,
                              n_out_range2_vec_static, n_particles,
                              n_measure,n_update_static, particles.maps_static ) ;
            for ( int n = 0 ; n < n_particles ; n++){
                REAL cn_update = sumGaussianMixtureWeights(particles.maps_static[n]) ;
                REAL update_val = evalGaussianMixture(particles.maps_static[n],
                                                     max_features[n].mean) ;
                REAL a = (1-config.pd)*config.clutterDensity*n_measure
                        + config.pd*n_measure*((n_measure-1)*config.clutterDensity*max_likelihoods[n]) ;
                REAL b = exp(cn_update - cn_predict[n] - config.clutterRate) ;
                REAL weight_factor = (a*predict_vals[n])/(b*update_val) ;
                particles.weights[n] += safeLog(weight_factor) ;
            }
        }
    }
    else if(config.featureModel==DYNAMIC_MODEL)
    {
        DEBUG_MSG("launching phdUpdateKernel Dynamic") ;
//        phdUpdateKernel<<<nBlocks,256>>>(
//            dev_poses, dev_maps_inrange_dynamic,dev_map_offsets_dynamic,
//            n_particles,n_measure,dev_maps_updated_dynamic,
//            dev_merged_flags_dynamic,dev_particle_weights) ;
        //
    }
    cudaPrintfDisplay(stdout,false) ;
    cudaPrintfEnd();

    checkCudaErrors( cudaFree( dev_maps_inrange_static ) ) ;
    checkCudaErrors( cudaFree( dev_map_offsets_static ) ) ;


//    // check input weights against merge flags
//    cout << "DEBUG first updated dynamic feature" << endl ;
//    bool* merged_flags = (bool*)malloc(n_update_dynamic*sizeof(bool)) ;
//    Gaussian4D* maps_updated = (Gaussian4D*)malloc( n_update_dynamic*sizeof(Gaussian4D) ) ;
//    cudaMemcpy( merged_flags, dev_merged_flags_dynamic, n_update_dynamic*sizeof(bool),cudaMemcpyDeviceToHost) ;
//    checkCudaErrors(
//                cudaMemcpy( maps_updated, dev_maps_updated_dynamic,
//                            n_update_dynamic*sizeof(Gaussian4D),
//                            cudaMemcpyDeviceToHost ) ) ;
//    for (int j = 0 ; j < n_update_dynamic ; j++)
//    {
//        cout << "(" << maps_updated[j].weight << " | " << merged_flags[j] << ")" << endl ;
//    }
//    print_feature(maps_updated[0]) ;
//    print_feature(maps_updated[1]) ;
//    free(maps_updated) ;
//    free(merged_flags) ;


    /******************************************************
     *
     * Merge updated maps and copy back to host
     *
     ******************************************************/
    if(config.featureModel==STATIC_MODEL || config.featureModel==MIXED_MODEL && config.particleWeighting != 2)
    {
        mergeAndCopyMaps( dev_maps_updated_static,dev_merged_flags_static,
                          features_out1_static,
                          features_out2_static, n_in_range_vec_static,
                          n_out_range1_vec_static,
                          n_out_range2_vec_static, n_particles,
                          n_measure,n_update_static, particles.maps_static ) ;
    }

    if(config.featureModel==DYNAMIC_MODEL || config.featureModel==MIXED_MODEL)
    {
        // TODO: hack to kill of out-of-range dynamic features
        features_out1_dynamic.clear();
        features_out2_dynamic.clear();
        n_out_range1_vec_dynamic.assign(n_particles,0);
        n_out_range2_vec_dynamic.assign(n_particles,0);
        mergeAndCopyMaps( dev_maps_updated_dynamic,dev_merged_flags_dynamic,
                          features_out1_dynamic,
                          features_out2_dynamic, n_in_range_vec_dynamic,
                          n_out_range1_vec_dynamic,
                          n_out_range2_vec_dynamic, n_particles,
                          n_measure,n_update_dynamic,particles.maps_dynamic ) ;
    }


    /**********************************************************
      *
      * Update particle weights
      *
      *********************************************************/
    DEBUG_MSG("Updating Particle Weights") ;
    if(config.particleWeighting != 2){
        REAL* particle_weights = (REAL*)malloc(n_particles*sizeof(REAL)) ;
        checkCudaErrors( cudaMemcpy(particle_weights,dev_particle_weights,
                                   n_particles*sizeof(REAL),
                                   cudaMemcpyDeviceToHost ) ) ;
        // multiply weights by multi-object likelihood
        for ( int i = 0 ; i < n_particles ; i++ )
        {
            particles.weights[i] += particle_weights[i]  ;
        }
        free(particle_weights) ;
    }

    // normalize
    REAL weightSum = logSumExp(particles.weights) ;
    DEBUG_VAL(weightSum) ;
    for (int i = 0 ; i < n_particles ; i++ )
    {
        particles.weights[i] -= weightSum ;
//        DEBUG_VAL(particles.weights[i]) ;
    }

    // free memory
    checkCudaErrors( cudaFree( dev_particle_weights ) ) ;
    checkCudaErrors( cudaFree( dev_poses ) ) ;
    return particlesPreMerge ;
}

//SmcPhdSLAM
//phdUpdate(SmcPhdSLAM& slam, measurementSet measurements)
//{
//    SmcPhdStatic maps_static_concat ;
//    SmcPhdDynamic maps_dynamic_concat ;
//    vector<int> map_sizes_static ;
//    vector<int> map_sizes_dynamic ;
//    // count map sizes
//    int n_particles = slam.n_particles ;
//    for (int n = 0 ; n < n_particles ; n++ )
//    {
//        map_sizes_static.push_back(slam.maps_static[n].x.size());
//        map_sizes_dynamic.push_back(slam.maps_dynamic[n].x.size());
//    }
//}

template <class GaussianType>
vector<GaussianType> computeExpectedMap(vector<vector <GaussianType> > maps,
                                        vector<REAL> weights)
// concatenate all particle maps into a single slam particle and then call the
// existing gaussian pruning algorithm ;
{
    DEBUG_MSG("Computing Expected Map") ;
    vector<GaussianType> concat ;
    int n_particles = maps.size() ;
    int* merged_sizes = (int*)malloc(n_particles*sizeof(int)) ;
    int* map_sizes = (int*)malloc(n_particles*sizeof(int)) ;
    int total_features = 0 ;
    for ( int n = 0 ; n < n_particles ; n++ )
    {
        vector<GaussianType> map = maps[n] ;
        for ( int i = 0 ; i < map.size() ; i++ )
            map[i].weight *= exp(weights[n]) ;
        concat.insert( concat.end(), map.begin(), map.end() ) ;
        merged_sizes[n] =  map.size() ;
        total_features += map.size() ;
    }

    if ( total_features == 0 )
    {
        DEBUG_MSG("no features") ;
        vector<GaussianType> expected_map(0) ;
        return expected_map ;
    }
    GaussianType* all_features = (GaussianType*)malloc( total_features*sizeof(GaussianType) ) ;
    std::copy( concat.begin(), concat.end(), all_features ) ;
    bool* merged_flags = (bool*)malloc( total_features*sizeof(sizeof(GaussianType) ) ) ;
    std::fill( merged_flags, merged_flags+total_features, false ) ;
    GaussianType* maps_out = (GaussianType*)malloc( total_features*sizeof(GaussianType) ) ;

    GaussianType* dev_maps_in = NULL ;
    GaussianType* dev_maps_out = NULL ;
    int* dev_merged_sizes = NULL ;
    bool* dev_merged_flags = NULL ;
    int* dev_map_sizes = NULL ;
    checkCudaErrors( cudaMalloc( (void**)&dev_maps_in,
                                total_features*sizeof(GaussianType) ) ) ;
    checkCudaErrors( cudaMalloc( (void**)&dev_maps_out,
                                total_features*sizeof(GaussianType) ) ) ;
    checkCudaErrors( cudaMalloc( (void**)&dev_merged_sizes,
                                n_particles*sizeof(int) ) ) ;
    checkCudaErrors( cudaMalloc( (void**)&dev_map_sizes,
                                n_particles*sizeof(int) ) ) ;
    checkCudaErrors( cudaMalloc( (void**)&dev_merged_flags,
                                total_features*sizeof(bool) ) ) ;
    for ( int n = n_particles/2 ; n > 0 ; n >>= 1 )
    {
        DEBUG_VAL(n) ;
        for ( int i = 0 ; i < n ; i++ )
            map_sizes[i] = merged_sizes[2*i] + merged_sizes[2*i+1] ;
        checkCudaErrors( cudaMemcpy( dev_map_sizes, map_sizes,
                                    n*sizeof(int),
                                    cudaMemcpyHostToDevice ) ) ;
        checkCudaErrors( cudaMemcpy( dev_maps_in, all_features,
                                    total_features*sizeof(GaussianType),
                                    cudaMemcpyHostToDevice) ) ;
        checkCudaErrors( cudaMemcpy( dev_merged_flags, merged_flags,
                                    total_features*sizeof(bool),
                                    cudaMemcpyHostToDevice)) ;
        // kernel launch
        phdUpdateMergeKernel<<<n,256>>>(
            dev_maps_in, dev_maps_out, dev_merged_sizes,
            dev_merged_flags, dev_map_sizes, n ) ;

        checkCudaErrors( cudaMemcpy( maps_out, dev_maps_out,
                                    total_features*sizeof(GaussianType),
                                    cudaMemcpyDeviceToHost) ) ;
        checkCudaErrors( cudaMemcpy( merged_sizes, dev_merged_sizes,
                                    n*sizeof(int), cudaMemcpyDeviceToHost ) ) ;
        int offset_in = 0 ;
        int offset_out = 0 ;
        for ( int i = 0 ; i < n ; i++ )
        {
            int n_copy = merged_sizes[i] ;
            std::copy( maps_out+offset_out, maps_out+offset_out+n_copy,
                       all_features+offset_in) ;
            offset_out += map_sizes[i] ;
            offset_in += n_copy ;
        }
        total_features = offset_in ;
    }
    vector<GaussianType> expected_map(total_features) ;
    std::copy( all_features,all_features+total_features, expected_map.begin() ) ;

    checkCudaErrors( cudaFree( dev_maps_in ) ) ;
    checkCudaErrors( cudaFree( dev_maps_out ) ) ;
    checkCudaErrors( cudaFree( dev_merged_sizes ) ) ;
    checkCudaErrors( cudaFree( dev_merged_flags ) ) ;
    checkCudaErrors( cudaFree( dev_map_sizes ) ) ;
    free(all_features) ;
    free(merged_flags) ;
    free(maps_out) ;
    return expected_map ;
}

template<class GaussianType>
bool expectedFeaturesPredicate( GaussianType g )
{
    return (g.weight <= config.minExpectedFeatureWeight) ;
}

/// copy the configuration structure to constant device memory
void
setDeviceConfig( const SlamConfig& config )
{
    checkCudaErrors(cudaMemcpyToSymbol( dev_config, &config, sizeof(SlamConfig) ) ) ;
//    seed_rng();
}

///////////////////////////////////////////////////////////////////

__host__ __device__ void
transformCameraToWorld(REAL xCamera, REAL yCamera, REAL zCamera,
                      CameraState cam,
                      REAL& xWorld, REAL& yWorld, REAL& zWorld,
                      bool isPoint=true){
    REAL croll = cos(cam.pose.proll) ;
    REAL cpitch = cos(cam.pose.ppitch) ;
    REAL cyaw = cos(cam.pose.pyaw) ;
    REAL sroll = sin(cam.pose.proll) ;
    REAL spitch = sin(cam.pose.ppitch) ;
    REAL syaw = sin(cam.pose.pyaw) ;

    xWorld = xCamera*(cpitch*cyaw) +
            yCamera*(croll*syaw + sroll*spitch*cyaw) +
            zCamera*(sroll*syaw - croll*spitch*cyaw) ;
    yWorld = xCamera*(-cpitch*syaw) +
            yCamera*(croll*cyaw - sroll*spitch*syaw) +
            zCamera*(sroll*cyaw + croll*spitch*syaw) ;
    zWorld = xCamera*(spitch) +
            yCamera*(-sroll*cpitch) +
            zCamera*(croll*cpitch) ;

    if(isPoint){
        xWorld += cam.pose.px ;
        yWorld += cam.pose.py ;
        zWorld += cam.pose.pz ;
    }
}

__host__ __device__ void
transformWorldToCamera(REAL xWorld, REAL yWorld, REAL zWorld,
                       CameraState cam,
                       REAL& xCamera, REAL& yCamera, REAL& zCamera,
                       bool isPoint=true){
    REAL croll = cos(cam.pose.proll) ;
    REAL cpitch = cos(cam.pose.ppitch) ;
    REAL cyaw = cos(cam.pose.pyaw) ;
    REAL sroll = sin(cam.pose.proll) ;
    REAL spitch = sin(cam.pose.ppitch) ;
    REAL syaw = sin(cam.pose.pyaw) ;
    xCamera = xWorld*(cpitch*cyaw) +
            yWorld*(-cpitch*syaw) +
            zWorld*(spitch) ;
    yCamera = xWorld*(croll*syaw + sroll*spitch*cyaw) +
            yWorld*(croll*cyaw - sroll*spitch*syaw) +
            zWorld*(-sroll*cpitch) ;
    zCamera = (xWorld)*(sroll*syaw - croll*spitch*cyaw) +
            (yWorld)*(sroll*cyaw + croll*spitch*syaw) +
            (zWorld)*(croll*cpitch) ;

    if(isPoint){
        xCamera += -cam.pose.px*(cpitch*cyaw) -
                cam.pose.py*(-cpitch*syaw) -
                cam.pose.pz*spitch ;
        yCamera += -cam.pose.px*(croll*syaw + sroll*spitch*cyaw) -
                cam.pose.py*(croll*cyaw - sroll*spitch*syaw) -
                cam.pose.pz*(-sroll*cpitch) ;
        zCamera += -cam.pose.px*(sroll*syaw - croll*spitch*cyaw) -
                cam.pose.py*(sroll*cyaw + croll*spitch*syaw) -
                cam.pose.pz*(croll*cpitch) ;
    }
}

/// functor for use with thrust::transform to convert particles in euclidean
/// space to particles in disparity space (baseline = 1)
/** pass a vector of camera states to the constructor
  * the argument to the functor is an 8-element tuple, where each element is a
  * vector with one element per feature particle. The first four elements are
    inputs:
  *     idx: index to the vector of camera states indicating to which camera
             this particle belongs
        x: x-coordinate of the particle
        y: y-coordinate of the particle
        z: z-coordinate of the particle
    The last 4 elements are outputs computed by the functor:
        u: u-coordinate of particle in disparity space
        v: v-coordinate of particle in disparity space
        d: disparity value of particle in disparity space
        out_of_range: 1 if the particle is not visible to the camera
  **/
struct world_to_disparity_transform{
    const CameraState* camera_states ;

    world_to_disparity_transform(CameraState* _states) : camera_states(_states) {}


    template <typename Tuple>
    __host__ __device__ void
    operator()(Tuple t){
        using namespace thrust ;
        CameraState cam = camera_states[get<0>(t)] ;
        REAL x = get<1>(t) ;
        REAL y = get<2>(t) ;
        REAL z = get<3>(t) ;

        REAL xCamera = 0 ;
        REAL yCamera = 0 ;
        REAL zCamera = 0 ;

        transformWorldToCamera(x,y,z,cam,xCamera,yCamera,zCamera) ;

        get<4>(t) = cam.u0 - cam.fx*xCamera/zCamera ;
        get<5>(t) = cam.v0 - cam.fy*yCamera/zCamera ;
        get<6>(t) = -cam.fx/zCamera ;

        bool in_fov = (get<4>(t) > 0) &&
                (get<4>(t) < dev_config.imageWidth) &&
                (get<5>(t) > 0) &&
                (get<5>(t) < dev_config.imageHeight) &&
                (get<6>(t) >= 0);
        get<7>(t) = in_fov ? 1 : 0 ;
    }
};

/// functor for use with thrust::transform to convert particles in disparity
/// space to particles in euclidean space (baseline = 1)
/** pass a vector of camera states to the constructor
  * the argument to the functor is an 7-element tuple, where each element is a
  * vector with one element per feature particle. The first four elements are
    inputs:
  *     idx: index to the vector of camera states indicating to which camera
             this particle belongs
        u: u-coordinate of particle in disparity space
        v: v-coordinate of particle in disparity space
        d: disparity value of particle in disparity space
    The last 3 elements are outputs computed by the functor:
        x: x-coordinate of the particle
        y: y-coordinate of the particle
        z: z-coordinate of the particle
  **/
struct disparity_to_world_transform{
    const CameraState* camera_states ;

    disparity_to_world_transform(CameraState* _states) : camera_states(_states) {}


    template <typename Tuple>
    __host__ __device__ void
    operator()(Tuple t){
        CameraState cam = camera_states[get<0>(t)] ;

        REAL u = get<1>(t) ;
        REAL v = get<2>(t) ;
        REAL d = get<3>(t) ;

        REAL xCamera = (u-cam.u0)/d ;
        REAL yCamera = cam.fx/cam.fy*(v-cam.v0)/d ;
        REAL zCamera = -cam.fx/d ;

        transformCameraToWorld(xCamera,yCamera,zCamera,cam,
                               get<4>(t),get<5>(t),get<6>(t)) ;
    }
};

/// this is a binary function which returns the sum of two numerical values
/// divided by an integer N. This can be used to compute the arithmetic mean of
/// N numbers by reduction.
struct compute_mean_function{
    const int N ;
    compute_mean_function(int _n) : N(_n) {}

    template <typename T>
    __host__ __device__ REAL
    operator()(T x, T y){
        return (REAL)(x+y)/(REAL)N ;
    }
};

/// unary operator which multiplies the argument by a constant
template <typename T>
struct multiply_by : public thrust::unary_function<T,T>
{
    const T N ;

    multiply_by(T _n) : N(_n) {}

    __host__ __device__ T
    operator()(T x){return x*N ;}
};


/// unary operator which divides the argument by a constant
template <typename T>
struct divide_by : public thrust::unary_function<T,T>
{
    const T N ;

    divide_by(T _n) : N(_n) {}

    __host__ __device__ T
    operator()(T x){return x/N ;}
};

/// unary operator which returns the weight of a gaussian object
template <typename T>
struct get_weight : public thrust::unary_function<T,REAL>
{
    __device__ REAL
    operator()(T g){ return g.weight; }
} ;

struct gt0 : public thrust::unary_function<REAL,bool>
{
    __host__ __device__ bool
    operator()(const REAL x){return (x>0);}
} ;

struct leq0 : public thrust::unary_function<REAL,bool>
{
    __host__ __device__ bool
    operator()(const REAL x){return (x<=0);}
} ;

// create predicate for testing feature visiblity
struct is_inrange : public thrust::unary_function<Gaussian3D,bool>
{
    __host__ __device__ bool
     operator()(const Gaussian3D g){
        REAL u = g.mean[0] ;
        REAL v = g.mean[1] ;
        REAL d = g.mean[2] ;

#if !defined(__CUDA_ARCH__)
        REAL width = config.imageWidth ;
        REAL height = config.imageHeight ;
#else
        REAL width = dev_config.imageWidth ;
        REAL height = dev_config.imageHeight ;
#endif

        bool in_fov = (u > 0) &&
            (u <= width) &&
            (v >= 0) &&
            (v <= height) &&
            (d >= 0);
        return in_fov ;
    }
};

__global__ void
fitGaussiansKernel(REAL* uArray, REAL* vArray, REAL* dArray,
                   REAL* weights,int nGaussians,
                   Gaussian3D* gaussians){
    int tid = threadIdx.x ;
    __shared__ REAL sdata[256] ;
    for (int i = blockIdx.x ; i < nGaussians ; i+=gridDim.x){
        int nParticles = dev_config.particlesPerFeature ;
        int offset = i*nParticles ;
        REAL val = 0 ;

        // compute mean u
        val = 0 ;
        for(int j = tid ; j < nParticles ; j+=blockDim.x){
            val += uArray[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        REAL uMean = sdata[0]/nParticles ;
        __syncthreads() ;

        // compute mean v
        val = 0 ;
        for(int j = tid ; j < nParticles ; j+=blockDim.x){
            val += vArray[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        REAL vMean = sdata[0]/nParticles ;
        __syncthreads() ;

        // compute mean d
        val = 0 ;
        for(int j = tid ; j < nParticles ; j+=blockDim.x){
            val += dArray[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        REAL dMean = sdata[0]/nParticles ;
        __syncthreads() ;


        // write means to output
        if (tid == 0){
//            cuPrintf("%f %f %f\n",uMean,vMean,dMean) ;
            gaussians[i].weight = weights[i] ;
            gaussians[i].mean[0] = uMean ;
            gaussians[i].mean[1] = vMean ;
            gaussians[i].mean[2] = dMean ;
        }

        // covariance term u-u
        val = 0 ;
        for(int j = tid ; j < nParticles ; j+=blockDim.x){
            val += pow(uArray[offset+j]-uMean,2) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0)
            gaussians[i].cov[0] = sdata[0]/(nParticles-1) ;
        __syncthreads() ;

        // covariance term v-v
        val = 0 ;
        for(int j = tid ; j < nParticles ; j+=blockDim.x){
            val += pow(vArray[offset+j]-vMean,2) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0)
            gaussians[i].cov[4] = sdata[0]/(nParticles-1) ;
        __syncthreads() ;

        // covariance term d-d
        val = 0 ;
        for(int j = tid ; j < nParticles ; j+=blockDim.x){
            val += pow(dArray[offset+j]-dMean,2) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0)
            gaussians[i].cov[8] = sdata[0]/(nParticles-1) ;
        __syncthreads() ;

        // covariance term u-v
        val = 0 ;
        for(int j = tid ; j < nParticles ; j+=blockDim.x){
            val += (uArray[offset+j]-uMean)*(vArray[offset+j]-vMean) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[1] = sdata[0]/(nParticles-1) ;
            gaussians[i].cov[3] = gaussians[i].cov[1] ;
        }
        __syncthreads() ;

        // covariance term u-d
        val = 0 ;
        for(int j = tid ; j < nParticles ; j+=blockDim.x){
            val += (uArray[offset+j]-uMean)*(dArray[offset+j]-dMean) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[2] = sdata[0]/(nParticles-1) ;
            gaussians[i].cov[6] = gaussians[i].cov[2] ;
        }
        __syncthreads() ;

        // covariance term v-d
        val = 0 ;
        for(int j = tid ; j < nParticles ; j+=blockDim.x){
            val += (vArray[offset+j]-vMean)*(dArray[offset+j]-dMean) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[5] = sdata[0]/(nParticles-1) ;
            gaussians[i].cov[7] = gaussians[i].cov[5] ;
        }
        __syncthreads() ;

    }
}

__global__ void
sampleGaussiansKernel(Gaussian3D* gaussians, int n_gaussians,
                      RngState* seeds,REAL* samples){
    int tid = blockIdx.x*blockDim.x + threadIdx.x ;

    if (tid < dev_config.particlesPerFeature){
        // initialize this thread's random number generator
        RngState random_state = seeds[tid] ;

        float x1,x2,x3,x_extra ;
        float2 randnorms ;
        bool odd_iteration = false ;

        int idx_result = tid ;
        int step = dev_config.particlesPerFeature*n_gaussians ;

        // loop over gaussians
        for (int n = 0 ; n < n_gaussians ; n++){
            // cholesky decomposition of covariance matrix
            REAL L11 = sqrt(gaussians[n].cov[0]) ;
            REAL L21 = gaussians[n].cov[1]/L11 ;
            REAL L22 = sqrt(gaussians[n].cov[4]-pow(L21,2)) ;
            REAL L31 = gaussians[n].cov[2]/L11 ;
            REAL L32 = (gaussians[n].cov[5]-L31*L21)/L22 ;
            REAL L33 = sqrt(gaussians[n].cov[8] - pow(L31,2) - pow(L32,2)) ;

            // generate uncorrelated normally distributed random values
            randnorms = randn(random_state) ;
            x1 = randnorms.x ;
            x2 = randnorms.y ;

            // the box-muller transform gives us normal variates two at a time,
            // but we only need 3, so on even iterations, we call the transform
            // twice and save the extra value to use in the next iteration.
            if ( !odd_iteration ){
                randnorms = randn(random_state) ;
                x3 = randnorms.x ;
                x_extra = randnorms.y ;
                odd_iteration = true ;
            }
            else
            {
                x3 = x_extra ;
                odd_iteration = false ;
            }

            // multiply uncorrelated values by cholesky decomposition and add
            // mean
            samples[idx_result] = x1*L11 + gaussians[n].mean[0] ;
            samples[idx_result+step] = x1*L21 + x2*L22 + gaussians[n].mean[1] ;
            samples[idx_result+2*step] = x1*L31 + x2*L32 + x3*L33 + gaussians[n].mean[2] ;
            idx_result += dev_config.particlesPerFeature ;
        }
    }
}

__global__ void
preUpdateDisparityKernel(Gaussian3D* features_predict,
                         REAL* features_pd,
                         int n_features,
                         ImageMeasurement* Z, int n_measure,
                         Gaussian3D* features_preupdate){
    int tid = blockIdx.x*blockDim.x + threadIdx.x ;
    for ( int i = tid ; i < n_features ; i+=gridDim.x*blockDim.x){
        Gaussian3D feature = features_predict[i] ;
        REAL pd = features_pd[i] ;

        // innovation covariance
        REAL sigma[4] ;
        REAL sigma_inv[4] ;
        REAL varU = pow(dev_config.stdU,2) ;
        REAL varV = pow(dev_config.stdV,2) ;
        sigma[0] = feature.cov[0] + varU ;
        sigma[1] = feature.cov[1] ;
        sigma[2] = feature.cov[3] ;
        sigma[3] = feature.cov[4] + varV ;
        invert_matrix2(sigma,sigma_inv) ;
        REAL det_sigma = sigma[0]*sigma[3] - sigma[1]*sigma[2] ;

        REAL K[6] ;
        K[0] = feature.cov[0]*sigma_inv[0] + feature.cov[3]*sigma_inv[1] ;
        K[1] = feature.cov[1]*sigma_inv[0] + feature.cov[4]*sigma_inv[1] ;
        K[2] = feature.cov[2]*sigma_inv[0] + feature.cov[5]*sigma_inv[1] ;
        K[3] = feature.cov[0]*sigma_inv[2] + feature.cov[3]*sigma_inv[3] ;
        K[4] = feature.cov[1]*sigma_inv[2] + feature.cov[4]*sigma_inv[3] ;
        K[5] = feature.cov[2]*sigma_inv[2] + feature.cov[5]*sigma_inv[3] ;

        // Maple-generated code for P = (IHK)*P*(IHK)' + KRK
        REAL cov_preupdate[9] ;
        cov_preupdate[0] = (1 - K[0]) * (feature.cov[0] * (1 - K[0]) - feature.cov[3] * K[3]) - K[3] * (feature.cov[1] * (1 - K[0]) - feature.cov[4] * K[3]) + varU *  pow( K[0],  2) + varV *  pow( K[3],  2);
        cov_preupdate[1] = -K[1] * (feature.cov[0] * (1 - K[0]) - feature.cov[3] * K[3]) + (1 - K[4]) * (feature.cov[1] * (1 - K[0]) - feature.cov[4] * K[3]) + K[0] * varU * K[1] + K[3] * varV * K[4];
        cov_preupdate[2] = -K[2] * (feature.cov[0] * (1 - K[0]) - feature.cov[3] * K[3]) - K[5] * (feature.cov[1] * (1 - K[0]) - feature.cov[4] * K[3]) + feature.cov[2] * (1 - K[0]) - feature.cov[5] * K[3] + K[0] * varU * K[2] + K[3] * varV * K[5];
        cov_preupdate[3] = (1 - K[0]) * (-feature.cov[0] * K[1] + feature.cov[3] * (1 - K[4])) - K[3] * (-feature.cov[1] * K[1] + feature.cov[4] * (1 - K[4])) + K[0] * varU * K[1] + K[3] * varV * K[4];
        cov_preupdate[4] = -K[1] * (-feature.cov[0] * K[1] + feature.cov[3] * (1 - K[4])) + (1 - K[4]) * (-feature.cov[1] * K[1] + feature.cov[4] * (1 - K[4])) + varU *  pow( K[1],  2) + varV *  pow( K[4],  2);
        cov_preupdate[5] = -K[2] * (-feature.cov[0] * K[1] + feature.cov[3] * (1 - K[4])) - K[5] * (-feature.cov[1] * K[1] + feature.cov[4] * (1 - K[4])) - feature.cov[2] * K[1] + feature.cov[5] * (1 - K[4]) + K[1] * varU * K[2] + K[4] * varV * K[5];
        cov_preupdate[6] = (1 - K[0]) * (-feature.cov[0] * K[2] - feature.cov[3] * K[5] + feature.cov[6]) - K[3] * (-feature.cov[1] * K[2] - feature.cov[4] * K[5] + feature.cov[7]) + K[0] * varU * K[2] + K[3] * varV * K[5];
        cov_preupdate[7] = -K[1] * (-feature.cov[0] * K[2] - feature.cov[3] * K[5] + feature.cov[6]) + (1 - K[4]) * (-feature.cov[1] * K[2] - feature.cov[4] * K[5] + feature.cov[7]) + K[1] * varU * K[2] + K[4] * varV * K[5];
        cov_preupdate[8] = -K[2] * (-feature.cov[0] * K[2] - feature.cov[3] * K[5] + feature.cov[6]) - K[5] * (-feature.cov[1] * K[2] - feature.cov[4] * K[5] + feature.cov[7]) - feature.cov[2] * K[2] - feature.cov[5] * K[5] + feature.cov[8] + varU *  pow( K[2],  2) + varV *  pow( K[5],  2);
        // end maple code

        for ( int m = 0 ; m < n_measure ; m++){
            int preupdate_idx = m*n_features + i ;
            REAL innov[2] ;
            innov[0] = Z[m].u - feature.mean[0] ;
            innov[1] = Z[m].v - feature.mean[1] ;

            REAL dist = innov[0]*innov[0]*sigma_inv[0] +
                    innov[0]*innov[1]*(sigma_inv[1]+sigma_inv[2]) +
                    innov[1]*innov[1]*sigma_inv[3] ;
            REAL log_weight = safeLog(pd) + safeLog(feature.weight)
                    - 0.5*dist - safeLog(2*M_PI) - 0.5*safeLog(det_sigma) ;

            features_preupdate[preupdate_idx].weight = log_weight ;
            features_preupdate[preupdate_idx].mean[0] = feature.mean[0] +
                    innov[0]*K[0] + innov[1]*K[3] ;
            features_preupdate[preupdate_idx].mean[1] = feature.mean[1] +
                    innov[0]*K[1] + innov[1]*K[4] ;
            features_preupdate[preupdate_idx].mean[2] = feature.mean[2] +
                    innov[0]*K[2] + innov[1]*K[5] ;
            for ( int n = 0 ; n < 9 ; n++ )
                features_preupdate[preupdate_idx].cov[n] = cov_preupdate[n] ;
        }
    }
}

/**
 * @brief separateDisparityFeatures Separate features into in-range and out-of-range parts
 * @param features_all[in] vector of disparity space gaussian features
 * @param offsets_all[in] indexing offsets to \p features_all
 * @param particles_all[in] vector of ParticleMaps corresponding to \p features_all
 * @param features_in[out] vector in-range disparity space gaussian features
 * @param offsets_in[out] indxing offsets to \p features_in
 * @param particles_out[out] vector of ParticleMaps containing the 3D particles
 *      for out-of-range features
 */
void separateDisparityFeatures(device_vector<Gaussian3D> features_all,
                    host_vector<int> offsets_all,
                    vector<ParticleMap> particles_all,
                    device_vector<Gaussian3D>& features_in,
                    host_vector<int>& offsets_in,
                    vector<ParticleMap>& particles_out)
{
    // make sure the output arrays are of sufficient size
    features_in.resize(features_all.size());
    offsets_in.resize(offsets_all.size());
    particles_out.resize(particles_all.size());

    // initialize the out-of-range particles to be empty
    for (int n = 0 ; n < particles_out.size() ; n++){
        particles_out[n].weights.clear();
        particles_out[n].x.clear();
        particles_out[n].y.clear();
        particles_out[n].z.clear();
    }

    // compute the in-range mask
    device_vector<bool> dev_inrange_mask(features_all.size()) ;
    DEBUG_MSG("transform") ;
    thrust::transform(features_all.begin(),features_all.end(),
                      dev_inrange_mask.begin(),is_inrange()) ;
    host_vector<bool> inrange_mask = dev_inrange_mask ;

    // do the separation
    DEBUG_MSG("copy_if") ;
    thrust::copy_if(features_all.begin(),features_all.end(),
                    features_in.begin(),
                    is_inrange()) ;

    if(config.debug){
        for ( int i = 0 ; i < inrange_mask.size() ; i++){
            std::cout << inrange_mask[i] << " " ;
            if (i % 20 == 0 && i > 0)
                std::cout << std::endl ;
        }
        std::cout << std::endl ;
    }

    // compute the separated offset arrays and copy out-of-range 3d particles
    int map_idx = 0 ;
    int feature_idx = 0 ;
    int start_particles = 0 ;
    int stop_particles = config.particlesPerFeature ;
    int offset_total = 0 ;
    DEBUG_MSG("compute offsets") ;
    for ( int i = 0 ; i < inrange_mask.size() ; i++ ){       
        // check if we have crossed over to the next map
        if( i >= offsets_all[map_idx+1] )
        {
            map_idx++ ;
            offsets_in[map_idx] = offset_total ;
            start_particles = 0 ;
            stop_particles = config.particlesPerFeature ;
            feature_idx = 0 ;
        }
//        DEBUG_VAL(start_particles) ;
//        DEBUG_VAL(stop_particles) ;
//        DEBUG_VAL(feature_idx) ;
//        DEBUG_VAL(map_idx) ;
//        DEBUG_VAL(inrange_mask[i]) ;
        if (inrange_mask[i])
        {
            offset_total++ ;
        }
        else{
            particles_out[map_idx].x.insert(particles_out[map_idx].x.end(),
                                            &particles_all[map_idx].x[start_particles],
                                            &particles_all[map_idx].x[stop_particles]) ;
            particles_out[map_idx].y.insert(particles_out[map_idx].y.end(),
                                            &particles_all[map_idx].y[start_particles],
                                            &particles_all[map_idx].y[stop_particles]) ;
            particles_out[map_idx].z.insert(particles_out[map_idx].z.end(),
                                            &particles_all[map_idx].z[start_particles],
                                            &particles_all[map_idx].z[stop_particles]) ;
            particles_out[map_idx].weights.push_back(particles_all[map_idx].weights[feature_idx]);
        }

        start_particles += config.particlesPerFeature ;
        stop_particles += config.particlesPerFeature ;
        feature_idx++ ;
    }
    map_idx++ ;
    offsets_in[map_idx] = offset_total ;

    // shrink the output arrays to fit data
    DEBUG_MSG("Shrink features_in") ;
    DEBUG_VAL(offsets_in.back()) ;
    features_in.resize(offsets_in.back());
    DEBUG_MSG("shrink_to_fit") ;
    features_in.shrink_to_fit();
}

/**
 * @brief recombineFeatures Merge in-range and out-of-range features into a
 *      single feature vector
 * @param features_in[in] vector of in-range features
 * @param offsets_in[in] vector of indexing offsets for in-range features
 * @param features_out[in] vector of out-of-range features
 * @param offsets_out[in] vector of indexing offsets for out-of-range features
 * @param features_all[out] vector where merged result will be written
 * @param offsets_all[out] indexing offsets for merged features
 */
void recombineFeatures(device_vector<Gaussian3D> features_in,
                       host_vector<int> offsets_in,
                       device_vector<Gaussian3D> features_out,
                       host_vector<int> offsets_out,
                       device_vector<Gaussian3D> features_all,
                       host_vector<int> offsets_all){

    // allocate space for outputs
    features_all.resize(features_in.size()+features_out.size());
    offsets_all.resize(offsets_in.size());
    device_vector<Gaussian3D>::iterator it_result = features_all.begin() ;

    // merge vectors map-by-map
    offsets_all[0] = 0 ;
    for ( int n = 0 ; n < offsets_in.size() ; n++ ){
        int start_in = offsets_in[n] ;
        int stop_in = offsets_in[n+1] ;
        it_result = thrust::copy(&features_in[start_in],
                          &features_in[stop_in],
                          it_result) ;
        int start_out = offsets_out[n] ;
        int stop_out = offsets_out[n+1] ;
        it_result = thrust::copy(&features_out[start_out],
                          &features_out[stop_out],
                          it_result) ;
        offsets_all[n+1] = stop_in + stop_out ;
    }
}

void
disparityPredict(DisparitySLAM& slam){
    DEBUG_MSG("Performing prediction") ;
    host_vector<CameraState> states = slam.states ;
    int n_states = states.size() ;
    vector<REAL> noise_x(n_states) ;
    vector<REAL> noise_y(n_states) ;
    vector<REAL> noise_z(n_states) ;
    vector<REAL> noise_roll(n_states) ;
    vector<REAL> noise_pitch(n_states) ;
    vector<REAL> noise_yaw(n_states) ;
    for (int i = 0 ; i < n_states ; i++){
        noise_x[i] = randn()*config.ax ;
        noise_y[i] = randn()*config.ay ;
        noise_z[i] = randn()*config.az ;
        noise_roll[i] = randn()*config.aroll ;
        noise_pitch[i] = randn()*config.apitch ;
        noise_yaw[i] = randn()*config.ayaw ;
    }
    REAL dt = config.dt ;
    for (int i = 0 ; i < n_states ; i++ ){
        ConstantVelocityState3D pose = slam.states[i].pose ;

        REAL dx = dt*pose.vx + 0.5*noise_x[i]*pow(dt,2) ;
        REAL dy = dt*pose.vy + 0.5*noise_y[i]*pow(dt,2) ;
        REAL dz = dt*pose.vz + 0.5*noise_z[i]*pow(dt,2) ;
        REAL dx_world = 0 ;
        REAL dy_world = 0 ;
        REAL dz_world = 0 ;
        transformCameraToWorld(dx,dy,dz,slam.states[i],
                               dx_world,dy_world,dz_world,false);

        pose.px += dx_world ;
        pose.py += dy_world ;
        pose.pz += dz_world ;
        pose.proll += dt*pose.vroll + 0.5*noise_roll[i]*pow(dt,2) ;
        pose.ppitch += dt*pose.vpitch + 0.5*noise_pitch[i]*pow(dt,2) ;
        pose.pyaw += dt*pose.vyaw + 0.5*noise_yaw[i]*pow(dt,2) ;
        pose.vx += dt*noise_x[i] ;
        pose.vy += dt*noise_y[i] ;
        pose.vz += dt*noise_z[i] ;
        pose.vroll += dt*noise_roll[i] ;
        pose.vpitch += dt*noise_pitch[i] ;
        pose.vyaw += dt*noise_yaw[i] ;

        pose.proll = wrapAngle(pose.proll) ;
        pose.ppitch = wrapAngle(pose.ppitch) ;
        pose.pyaw = wrapAngle(pose.pyaw) ;
        slam.states[i].pose = pose ;
    }
}

void
disparityUpdate(DisparitySLAM& slam,
                std::vector<ImageMeasurement> Z){

    host_vector<ImageMeasurement> measurements = Z ;
//    DEBUG_MSG("Received measurements: ") ;
//    for ( int i = 0 ; i < Z.size() ; i++ ){
//        cout << Z[i].u << "," << Z[i].v << endl ;
//    }

    // vector which contains the camera state to which each particle belongs
    host_vector<int> camera_idx_vector ;

    // vectors to contain concatenated particles
    host_vector<REAL> x_vector ;
    host_vector<REAL> y_vector ;
    host_vector<REAL> z_vector ;

    // vector to contain camera states
    host_vector<CameraState> camera_vector = slam.states ;
    if(config.debug){
        DEBUG_MSG("Camera states: ") ;
        for ( int n = 0 ; n < camera_vector.size() ; n++ ){
            CameraState cam = camera_vector[n] ;
            cout << n << " " << cam.pose.px << ","
                 << cam.pose.py << ","
                 << cam.pose.pz << ","
                 << cam.pose.proll << ","
                 << cam.pose.ppitch << ","
                 << cam.pose.pyaw << endl ;
        }
    }

    // vector of map sizes
    host_vector<int> map_offsets(slam.n_particles+1) ;
    map_offsets[0] = 0 ;

    // vector feature weights
    host_vector<REAL> feature_weights ;

    for ( int n = 0 ; n < slam.n_particles ; n++ ) {
        ParticleMap map_n = slam.maps[n] ;
        int n_particles = map_n.x.size() ;
        map_offsets[n+1] = map_offsets[n] + n_particles/config.particlesPerFeature ;
        camera_idx_vector.insert(camera_idx_vector.end(),
                               n_particles,
                               n) ;
        x_vector.insert(x_vector.end(),map_n.x.begin(),map_n.x.end()) ;
        y_vector.insert(y_vector.end(),map_n.y.begin(),map_n.y.end()) ;
        z_vector.insert(z_vector.end(),map_n.z.begin(),map_n.z.end()) ;

        feature_weights.insert(feature_weights.end(),
                               map_n.weights.begin(),
                               map_n.weights.end()) ;
    }

    // create device vectors
    int n_particles_total = x_vector.size() ;
    int n_features_total = n_particles_total/config.particlesPerFeature ;
    device_vector<CameraState> dev_camera_vector = camera_vector ;
    device_vector<int> dev_camera_idx_vector = camera_idx_vector ;
    device_vector<REAL> dev_x_vector = x_vector ;
    device_vector<REAL> dev_y_vector = y_vector ;
    device_vector<REAL> dev_z_vector = z_vector ;
    device_vector<REAL> dev_u_vector(n_particles_total) ;
    device_vector<REAL> dev_v_vector(n_particles_total) ;
    device_vector<REAL> dev_d_vector(n_particles_total) ;
    device_vector<REAL> dev_inrange_vector(n_particles_total) ;


    // do the transformation
    DEBUG_MSG("Performing world to disparity transformation") ;
    thrust::for_each(make_zip_iterator(make_tuple(
                                   dev_camera_idx_vector.begin(),
                                   dev_x_vector.begin(),
                                   dev_y_vector.begin(),
                                   dev_z_vector.begin(),
                                   dev_u_vector.begin(),
                                   dev_v_vector.begin(),
                                   dev_d_vector.begin(),
                                   dev_inrange_vector.begin()
                                   )),
             make_zip_iterator(make_tuple(
                                    dev_camera_idx_vector.end(),
                                    dev_x_vector.end(),
                                    dev_y_vector.end(),
                                    dev_z_vector.end(),
                                    dev_u_vector.end(),
                                    dev_v_vector.end(),
                                    dev_d_vector.end(),
                                    dev_inrange_vector.end()
                                    )),
             world_to_disparity_transform(raw_pointer_cast(&dev_camera_vector[0]))) ;


//    DEBUG_MSG("Disparity-transformed particles: ") ;
//    DEBUG_MSG("First map: ") ;
//    for( int j = 0 ; j < slam.maps[0].x.size() ; j++ ){
//        cout << dev_u_vector[j] << ","
//             << dev_v_vector[j] << ","
//             << dev_d_vector[j] << endl ;
//    }
//    DEBUG_MSG("Second map: ") ;
//    for( int j = 0 ; j < slam.maps[1].x.size() ; j++ ){
//        cout << dev_u_vector[j+slam.maps[0].x.size()] << ","
//             << dev_v_vector[j+slam.maps[0].x.size()] << ","
//             << dev_d_vector[j+slam.maps[0].x.size()] << endl ;
//    }

//    // generate the keys for grouping particles into features
//    host_vector<int> feature_keys ;
//
//    DEBUG_VAL(n_features_total) ;
//    for ( int n = 0 ; n < n_features_total ; n++ ){
//        feature_keys.insert(feature_keys.end(),config.particlesPerFeature,n) ;
//    }

//    // compute pd for each gaussian feature
//    DEBUG_MSG("Computing Pd") ;
//    device_vector<int> dev_feature_keys = feature_keys ;
//    device_vector<int> dev_keys_out(n_particles_total) ;
//    device_vector<REAL> dev_pd(n_particles_total) ;
//    // sum the in-range values of all particles per feature
//    reduce_by_key(dev_feature_keys.begin(),dev_feature_keys.end(),
//                  dev_inrange_vector.begin(),
//                  dev_keys_out.begin(),dev_pd.begin()) ;

//    // divide the sum by the number of particles per feature
//    divide_by<REAL> division_op((REAL)config.particlesPerFeature) ;
//    thrust::transform(dev_pd.begin(),dev_pd.end(),dev_pd.begin(),division_op) ;

//    // multiply by nominal pd value
//    multiply_by<REAL> multiply_op(config.pd) ;
//    thrust::transform(dev_pd.begin(),dev_pd.end(),dev_pd.begin(),multiply_op) ;

//    if(config.debug){
//        DEBUG_MSG("Computed Pd for first particle:") ;
//        for ( int j = 0 ; j < slam.maps[0].weights.size() ; j++ ){
//            cout << dev_pd[j] << endl ;
//        }
//    }

//    if (n_particles_total > 0 && config.debug){
//        DEBUG_MSG("Verify disparity space particles: ");
//        for (int j = 0 ; j < config.particlesPerFeature ; j++){
//            cout << dev_u_vector[j] << "," << dev_v_vector[j] << "," << dev_d_vector[j] << endl ;
//        }
//    }

//    DEBUG_MSG("Separate in-range and outside-range features") ;
//    int k = 0 ;
//    host_vector<REAL> pd_vector = dev_pd ;
//    device_vector<REAL> dev_u_inrange ;
//    host_vector<REAL> u_outrange ;
//    device_vector<REAL> dev_v_inrange ;
//    host_vector<REAL> v_outrange ;
//    device_vector<REAL> dev_d_inrange ;
//    host_vector<REAL> d_outrange ;
//    host_vector<int> map_offsets_inrange(slam.n_particles+1,0) ;
//    host_vector<int> map_offsets_outrange(slam.n_particles+1,0) ;
//    int n_features_inrange = 0 ;
//    DEBUG_MSG("particles...") ;
//    for (int i = 0 ; i < slam.n_particles ; i++ ){
//        int n_features = map_offsets[i+1]-map_offsets[i] ;
//        map_offsets_inrange[i+1] = map_offsets_inrange[i] ;
//        map_offsets_outrange[i+1] = map_offsets_outrange[i] ;
//        for ( int j = 0 ; j < n_features ; j++ ){
//            int offset_begin = k*config.particlesPerFeature ;
//            int offset_end = offset_begin + config.particlesPerFeature ;
//            if(pd_vector[k] > 0){
//                dev_u_inrange.insert(dev_u_inrange.end(),
//                                     dev_u_vector.begin()+offset_begin,
//                                     dev_u_vector.begin()+offset_end) ;
//                dev_v_inrange.insert(dev_v_inrange.end(),
//                                     dev_v_vector.begin()+offset_begin,
//                                     dev_v_vector.begin()+offset_end) ;
//                dev_d_inrange.insert(dev_d_inrange.end(),
//                                     dev_d_vector.begin()+offset_begin,
//                                     dev_d_vector.begin()+offset_end) ;
//                map_offsets_inrange[i+1]++ ;
//                n_features_inrange++ ;
//            }
//            else{
//                u_outrange.insert(u_outrange.end(),
//                                     dev_u_vector.begin()+offset_begin,
//                                     dev_u_vector.begin()+offset_end) ;
//                v_outrange.insert(v_outrange.end(),
//                                     dev_v_vector.begin()+offset_begin,
//                                     dev_v_vector.begin()+offset_end) ;
//                d_outrange.insert(d_outrange.end(),
//                                     dev_d_vector.begin()+offset_begin,
//                                     dev_d_vector.begin()+offset_end) ;
//                map_offsets_outrange[i+1]++ ;
//            }
//            k++ ;
//        }
//    }
//    DEBUG_MSG("weights...") ;
//    host_vector<REAL> feature_weights_inrange(n_features_total) ;
//    host_vector<REAL> feature_weights_outrange(n_features_total) ;
//    host_vector<REAL> pd_inrange(n_features_total) ;

//    DEBUG_MSG("copy inrange weights...") ;
//    thrust::copy_if(feature_weights.begin(),
//                    feature_weights.end(),
//                    pd_vector.begin(),
//                    feature_weights_inrange.begin(),
//                    gt0()) ;
//    DEBUG_MSG("copy outrange weights...") ;
//    thrust::copy_if(feature_weights.begin(),
//                    feature_weights.end(),
//                    pd_vector.begin(),
//                    feature_weights_outrange.begin(),
//                    leq0()) ;
//    DEBUG_MSG("copy pd in range") ;
//    thrust::copy_if(pd_vector.begin(),
//                    pd_vector.end(),
//                    pd_inrange.begin(),
//                    gt0()) ;
//    dev_pd = pd_inrange ;

    // fit gaussians to particles
    cudaPrintfInit() ;
    DEBUG_MSG("Fitting gaussians to disparity space particles") ;
    int n_blocks = min(65535,n_features_total) ;
    device_vector<REAL> dev_feature_weights = feature_weights ;
    device_vector<Gaussian3D> dev_gaussians(n_features_total) ;
    fitGaussiansKernel<<<n_blocks,256>>>
        (raw_pointer_cast(&dev_u_vector[0]),
         raw_pointer_cast(&dev_v_vector[0]),
         raw_pointer_cast(&dev_d_vector[0]),
         raw_pointer_cast(&dev_feature_weights[0]),
         n_features_total,
         raw_pointer_cast(&dev_gaussians[0]) ) ;
//    cudaPrintfDisplay() ;

    if(config.debug){
        DEBUG_MSG("Fitted gaussians:") ;
        for ( int n = 0 ; n < n_features_total ; n++ ){
            Gaussian3D g = dev_gaussians[n] ;
            print_feature(g) ;
        }
    }

    // separate in range and out of range gaussians
    DEBUG_MSG("Separating in-range features") ;
//    for(int n = 0 ; n < map_offsets.size() ; n++)
//        DEBUG_VAL(map_offsets[n]) ;
    host_vector<int> map_offsets_in(slam.n_particles+1,0) ;
    device_vector<Gaussian3D> dev_gaussians_in ;
    vector<ParticleMap> particles_out = slam.maps ;
    separateDisparityFeatures(dev_gaussians,map_offsets,slam.maps,
                     dev_gaussians_in,map_offsets_in,
                     particles_out);
    int n_features_in = map_offsets_in.back() ;
//    for(int n = 0 ; n < map_offsets_in.size() ; n++)
//        DEBUG_VAL(map_offsets_in[n]) ;

    device_vector<REAL> dev_pd(dev_gaussians_in.size(),config.pd) ;

//    if(config.debug){
//        DEBUG_MSG("in-range gaussians:") ;
//        for ( int n = 0 ; n < n_features_in ; n++ ){
//            Gaussian3D g = dev_gaussians_in[n] ;
//            print_feature(g) ;
//        }
//    }

//    if (config.debug){
//        DEBUG_MSG("out-of-range particles:") ;
//        particles_out[0].print() ;
//    }

    // generate the birth terms
    DEBUG_MSG("Generating birth terms from measurements") ;
    int n_measurements = measurements.size() ;
    host_vector<Gaussian3D> gaussians_birth(n_measurements*slam.n_particles) ;
    for ( int m = 0 ; m < n_measurements*slam.n_particles ; m++ ){
        if ( m < n_measurements ){
            gaussians_birth[m].weight = safeLog(config.birthWeight) ;
            gaussians_birth[m].mean[0] = measurements[m].u ;
            gaussians_birth[m].mean[1] = measurements[m].v ;
            gaussians_birth[m].mean[2] = config.disparityBirth ;
            gaussians_birth[m].cov[0] = pow(config.stdU,2) ;
            gaussians_birth[m].cov[4] = pow(config.stdV,2) ;
            gaussians_birth[m].cov[8] = pow(config.stdDBirth,2) ;
            gaussians_birth[m].cov[1] = 0 ;
            gaussians_birth[m].cov[2] = 0 ;
            gaussians_birth[m].cov[3] = 0 ;
            gaussians_birth[m].cov[5] = 0 ;
            gaussians_birth[m].cov[6] = 0 ;
            gaussians_birth[m].cov[7] = 0 ;
        }
        else
        {
            int idx = m % n_measurements ;
            copy_gaussians(gaussians_birth[idx],gaussians_birth[m]) ;
        }
//        print_feature(gaussians_birth[m]) ;
    }
    DEBUG_MSG("copy births to device") ;
    device_vector<Gaussian3D> dev_gaussians_birth = gaussians_birth ;

    DEBUG_VAL(n_measurements) ;
    if (config.debug){
        for ( int i = 0 ; i < measurements.size() ; i++){
            std::cout << measurements[i].u << "," << measurements[i].v << std::endl ;
        }
    }
    device_vector<ImageMeasurement> dev_measurements(n_measurements) ;
    DEBUG_MSG("copy measurements to device") ;
    dev_measurements = measurements ;

    // do the preupdate
    DEBUG_MSG("allocate preupdate terms") ;
    device_vector<Gaussian3D> dev_gaussians_preupdate(n_features_in*n_measurements) ;
    if (dev_gaussians_preupdate.size() > 0){
        DEBUG_MSG("Computing disparity pre-update") ;
        n_blocks = min(65535,n_features_in) ;
        preUpdateDisparityKernel<<<n_blocks,256>>>
            (raw_pointer_cast(&dev_gaussians_in[0]),
             raw_pointer_cast(&dev_pd[0]),
            n_features_in,
            raw_pointer_cast(&dev_measurements[0]),
            n_measurements,
            raw_pointer_cast(&dev_gaussians_preupdate[0]));
        if (config.debug){
            DEBUG_MSG("pre-update terms:") ;
            for(int j = 0 ; j < n_features_in*n_measurements ; j++ ){
                Gaussian3D g= dev_gaussians_preupdate[j] ;
                print_feature(g) ;
            }
        }
    }

    // do the sc-phd update
    DEBUG_VAL(config.birthWeight) ;
    DEBUG_VAL(config.clutterDensity) ;
    DEBUG_MSG("allocate particle weights") ;
    device_vector<REAL> dev_weights(slam.n_particles) ;
    DEBUG_MSG("copy map offsets to device") ;
    device_vector<int> dev_map_offsets = map_offsets_in ;
    int n_update = n_features_in*(n_measurements+1) +
            slam.n_particles*n_measurements ;
    DEBUG_VAL(n_update) ;
    DEBUG_MSG("allocate device memory for updated gaussians") ;
    device_vector<Gaussian3D> dev_gaussians_update(n_update) ;
    DEBUG_MSG("allocate device memory for merging flags") ;
    device_vector<bool> dev_merge_flags(n_update) ;
    n_blocks = min(slam.n_particles,65535) ;

//    for ( int n = 0 ; n < slam.n_particles ; n++){
//        int x = dev_map_offsets[n] ;
//        DEBUG_VAL(x) ;
//    }

    DEBUG_MSG("Performing SC-PHD update") ;
    phdUpdateKernel<<<n_blocks,256>>>
        (raw_pointer_cast(&dev_gaussians_in[0]),
         raw_pointer_cast(&dev_pd[0]),
         raw_pointer_cast(&dev_gaussians_preupdate[0]),
         raw_pointer_cast(&dev_gaussians_birth[0]),
         raw_pointer_cast(&dev_map_offsets[0]),
         slam.n_particles,n_measurements,
         raw_pointer_cast(&dev_gaussians_update[0]),
         raw_pointer_cast(&dev_merge_flags[0]),
         raw_pointer_cast(&dev_weights[0])) ;

    //
//    cudaPrintfDisplay() ;
    cudaPrintfEnd();

    // manually free some device memory
    dev_gaussians_birth.resize(0);
    dev_gaussians_birth.shrink_to_fit();

    dev_gaussians.resize(0);
    dev_gaussians.shrink_to_fit();

    dev_gaussians_preupdate.resize(0);
    dev_gaussians_preupdate.shrink_to_fit();

    if(config.debug){
        DEBUG_MSG("Updated gaussians and merge flags: ") ;
        for (int n = 0 ; n < n_update ; n++){
            bool flag = dev_merge_flags[n] ;
            cout << flag << " " ;
            Gaussian3D g = dev_gaussians_update[n] ;
            print_feature(g) ;
        }
    }

    // do the GM-merging
    device_vector<int> dev_merged_sizes(slam.n_particles) ;
    device_vector<Gaussian3D> dev_gaussians_merged_tmp(n_update) ;

    // recalculate offsets for updated map size
    for ( int n = 0 ; n < (slam.n_particles+1) ; n++ ){
        map_offsets_in[n] *= (n_measurements+1) ;
        map_offsets_in[n] += n_measurements*n ;
//        DEBUG_VAL(map_offsets[n]) ;
    }
    dev_map_offsets = map_offsets_in ;

    DEBUG_MSG("Performing GM reduction") ;
    phdUpdateMergeKernel<<<n_blocks,256>>>
     (raw_pointer_cast(&dev_gaussians_update[0]),
      raw_pointer_cast(&dev_gaussians_merged_tmp[0]),
      raw_pointer_cast(&dev_merged_sizes[0]),
      raw_pointer_cast(&dev_merge_flags[0]),
      raw_pointer_cast(&dev_map_offsets[0]),
      slam.n_particles) ;
    //

    // copy out the results of the GM reduction, leaving only valid gaussians
    host_vector<int> merged_sizes = dev_merged_sizes ;
    int n_merged_total = thrust::reduce(merged_sizes.begin(),
                                        merged_sizes.end()) ;
    device_vector<Gaussian3D> dev_gaussians_merged(n_merged_total) ;
    device_vector<Gaussian3D>::iterator it = dev_gaussians_merged.begin() ;
    for ( int n = 0 ; n < merged_sizes.size() ; n++){
        it = thrust::copy_n(&dev_gaussians_merged_tmp[map_offsets_in[n]],
                        merged_sizes[n],
                        it) ;
    }

    // get the updated feature weights
    device_vector<REAL> dev_merged_weights(n_merged_total) ;
    get_weight<Gaussian3D> op ;
    thrust::transform(dev_gaussians_merged.begin(),
                      dev_gaussians_merged.end(),
                      dev_merged_weights.begin(),
                      op) ;
    host_vector<REAL> merged_weights = dev_merged_weights ;
    if (config.debug)
    {
        DEBUG_MSG("merged feature weights: ") ;
        for( int n = 0 ; n < merged_weights.size() ; n++){
            cout << merged_weights[n] << endl ;
        }
    }

    // initialize seeds for device-side random number generators
    host_vector<RngState> seeds(config.particlesPerFeature) ;
    for ( int n = 0 ; n < config.particlesPerFeature ; n++ ){
        seeds[n].z1 = static_cast<unsigned>(randu01()*127 + 129) ;
        seeds[n].z2 = static_cast<unsigned>(randu01()*127 + 129) ;
        seeds[n].z3 = static_cast<unsigned>(randu01()*127 + 129) ;
        seeds[n].z4 = static_cast<unsigned>(randu01()*256) ;
    }
    device_vector<RngState> dev_seeds = seeds ;
//    DEBUG_MSG("seeds: ") ;
//    for (int n = 0 ; n < seeds.size() ; n++){
//        cout << "[" << seeds[n].z1 << "," << seeds[n].z2 << ","
//             << seeds[n].z3 << "," << seeds[n].z4 << "]" << endl ;
//    }

    // generate samples from merged gaussians
    DEBUG_MSG("Sampling merged gaussians") ;
    int n_particles_merged = n_merged_total*config.particlesPerFeature ;
    device_vector<REAL> dev_samples(3*n_particles_merged) ;
    n_blocks = ceil(config.particlesPerFeature/256.0) ;
    DEBUG_VAL(n_blocks) ;
    sampleGaussiansKernel<<<n_blocks,256>>>(
                            raw_pointer_cast(&dev_gaussians_merged[0]),
                            n_merged_total,
                            raw_pointer_cast(&dev_seeds[0]),
                            raw_pointer_cast(&dev_samples[0]));

    if(config.debug){
        DEBUG_MSG("Verify Gaussian sampling:") ;
        Gaussian3D g = dev_gaussians_merged[0] ;
        print_feature(g) ;
        for(int j = 0 ; j < config.particlesPerFeature ; j++){
            cout << dev_samples[j] << ","
                 << dev_samples[j+n_particles_merged] << ","
                 << dev_samples[j+2*n_particles_merged] << endl ;
        }
    }

    // split samples into individual components
    dev_u_vector.resize(n_particles_merged);
    dev_v_vector.resize(n_particles_merged);
    dev_d_vector.resize(n_particles_merged);
    thrust::copy_n(dev_samples.begin(),
                   n_particles_merged,dev_u_vector.begin()) ;
    thrust::copy_n(dev_samples.begin()+n_particles_merged,
                   n_particles_merged,dev_v_vector.begin()) ;
    thrust::copy_n(dev_samples.begin()+2*n_particles_merged,
                   n_particles_merged,dev_d_vector.begin()) ;


    // prepare the camera index vector for transforming the particles
    // and save gaussian weights
    camera_idx_vector.clear();
    int offset = 0 ;
    for ( int n = 0 ; n < slam.n_particles ; n++ ){
        int n_merged = merged_sizes[n] ;
        camera_idx_vector.insert(camera_idx_vector.end(),
                                 n_merged*config.particlesPerFeature, n) ;
        slam.maps[n].weights.assign(&merged_weights[offset],
                                    &merged_weights[offset+n_merged]);
        offset += n_merged ;
    }
    dev_camera_idx_vector = camera_idx_vector ;


//    // copy merged features to host, and sample disparity particles
//    DEBUG_MSG("Sampling disparity space particles") ;

//    host_vector<REAL> u_vector ;
//    host_vector<REAL> v_vector ;
//    host_vector<REAL> d_vector ;
//    host_vector<Gaussian3D> gaussians_merged = dev_gaussians_merged ;
//    camera_idx_vector.clear();
//    for ( int n = 0 ; n < slam.n_particles ; n++ ){
//        int offset = map_offsets_in[n] ;
//        int n_merged = merged_sizes[n] ;
//        if(config.debug)
//            DEBUG_VAL(n_merged) ;
//        host_vector<REAL> weights(0) ;
//        camera_idx_vector.insert(camera_idx_vector.end(),
//                               n_merged*config.particlesPerFeature, n) ;
//        for ( int i = 0 ; i < n_merged ; i++ ){
//            Gaussian3D g = gaussians_merged[offset+i] ;
////            if(config.debug)
////                print_feature(g) ;
//            vector<REAL> samples(config.particlesPerFeature*3) ;
//            randmvn3(g.mean,g.cov,config.particlesPerFeature,&samples[0]);
//            REAL* u_ptr = &samples[0] ;
//            REAL* v_ptr = u_ptr+config.particlesPerFeature ;
//            REAL* d_ptr = v_ptr+config.particlesPerFeature ;
//            u_vector.insert(u_vector.end(),
//                            u_ptr, u_ptr+config.particlesPerFeature) ;
//            v_vector.insert(v_vector.end(),
//                            v_ptr, v_ptr+config.particlesPerFeature) ;
//            d_vector.insert(d_vector.end(),
//                            d_ptr, d_ptr+config.particlesPerFeature) ;

//            // save the gaussian weight now
//            weights.push_back(g.weight);
//        }
//        slam.maps[n].weights.assign(weights.begin(),weights.end()) ;
//    }


    // copy disparity particles to device
//    n_particles_total = u_vector.size() ;
//    dev_u_vector = u_vector ;
//    dev_v_vector = v_vector ;
//    dev_d_vector = d_vector ;
    dev_x_vector.resize(n_particles_merged);
    dev_y_vector.resize(n_particles_merged);
    dev_z_vector.resize(n_particles_merged);


//    for (int n = 0 ; n < u_vector.size() ; n++ )
//        DEBUG_VAL(u_vector[n]) ;

    // do the transformation
    DEBUG_MSG("Computing disparity to world transformation") ;
    thrust::for_each(make_zip_iterator(make_tuple(
                                   dev_camera_idx_vector.begin(),
                                   dev_u_vector.begin(),
                                   dev_v_vector.begin(),
                                   dev_d_vector.begin(),
                                   dev_x_vector.begin(),
                                   dev_y_vector.begin(),
                                   dev_z_vector.begin()
                                   )),
             make_zip_iterator(make_tuple(
                                dev_camera_idx_vector.end(),
                                dev_u_vector.end(),
                                dev_v_vector.end(),
                                dev_d_vector.end(),
                                dev_x_vector.end(),
                                dev_y_vector.end(),
                                dev_z_vector.end()
                                )),
             disparity_to_world_transform(raw_pointer_cast(&dev_camera_vector[0]))) ;

    // save euclidean particles
    DEBUG_MSG("Saving updated 3D particles") ;
    x_vector = dev_x_vector ;
    y_vector = dev_y_vector ;
    z_vector = dev_z_vector ;
    host_vector<REAL> weights = dev_weights ;

    if(config.debug){
        DEBUG_MSG("Verify disparity to euclidean transformation") ;
        for( int j = 0 ; j < config.particlesPerFeature ; j++ ){
            cout << x_vector[j] << "," << y_vector[j] << "," << z_vector[j] << endl ;
        }
    }

    offset = 0 ;
    for ( int n = 0 ; n < slam.n_particles ; n++ ){
//        DEBUG_VAL(slam.weights[n]) ;
        int n_particles = merged_sizes[n]*config.particlesPerFeature ;
        slam.maps[n].x.assign(x_vector.begin()+offset,
                              x_vector.begin()+offset+n_particles) ;
        slam.maps[n].y.assign(y_vector.begin()+offset,
                              y_vector.begin()+offset+n_particles) ;
        slam.maps[n].z.assign(z_vector.begin()+offset,
                              z_vector.begin()+offset+n_particles) ;
        offset += n_particles ;

        // recombine with out of range particles
        slam.maps[n].weights.insert(slam.maps[n].weights.end(),
                                    particles_out[n].weights.begin(),
                                    particles_out[n].weights.end()) ;
        slam.maps[n].x.insert(slam.maps[n].x.end(),
                              particles_out[n].x.begin(),
                              particles_out[n].x.end()) ;
        slam.maps[n].y.insert(slam.maps[n].y.end(),
                              particles_out[n].y.begin(),
                              particles_out[n].y.end()) ;
        slam.maps[n].z.insert(slam.maps[n].z.end(),
                              particles_out[n].z.begin(),
                              particles_out[n].z.end()) ;

        // update parent particle weights
        slam.weights[n] += weights[n] ;
        if (config.debug)
            DEBUG_VAL(slam.weights[n]) ;
    }

//    if (config.debug){
//        DEBUG_MSG("Updated map particles: ") ;
//        for ( int n = 0 ; n < slam.n_particles ; n++ ){
//            DEBUG_VAL(n) ;
//            slam.maps[n].print() ;
//        }
//    }

    // normalize particle weights
    DEBUG_MSG("normalize weights") ;
    REAL log_weight_sum = logSumExp(slam.weights) ;
    DEBUG_VAL(log_weight_sum) ;
    for(int n = 0 ; n < slam.n_particles ; n++ ){
        slam.weights[n] -= log_weight_sum ;
        if(config.debug)
            DEBUG_VAL(slam.weights[n]) ;
    }
}
