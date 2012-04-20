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
#include <cutil.h>
#include <complex.h>
#include <fftw3.h>
#include <assert.h>
#include <float.h>
#include "cuPrintf.cu"

#include "device_math.cuh"

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
predictMap(ParticleSLAM& p) ;

void
phdPredict(ParticleSLAM& particles, ... ) ;

template<class GaussianType>
void
phdPredictVp( ParticleSLAM& particles ) ;

ParticleSLAM
phdUpdate(ParticleSLAM& particles, measurementSet measurements) ;

ParticleSLAM
resampleParticles( ParticleSLAM oldParticles, int n_particles=-1 ) ;

void
recoverSlamState(ParticleSLAM particles, ConstantVelocityState& expectedPose,
        vector<Gaussian2D>& expectedMap, vector<Gaussian4D>& expectedMapDynamic,
                 vector<REAL>& cn_estimate ) ;

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

// Constant memory variables
__device__ __constant__ RangeBearingMeasurement Z[256] ;
__device__ __constant__ SlamConfig dev_config ;

// other global device variables
REAL* dev_C ;
REAL* dev_factorial ;
REAL* log_factorials ;
//__device__ REAL* dev_qspower ;
//__device__ REAL* dev_pspower ;
REAL* dev_cn_clutter ;

//ConstantVelocityModelProps modelProps  = {STDX, STDY,STDTHETA} ;
//ConstantVelocity2DKinematicModel motionModel(modelProps) ;

/// helper function for outputting a Gaussian to std_out
template<class GaussianType>
__host__ __device__ void
print_feature(GaussianType f)
{
    int dims = getGaussianDim(f) ;
#ifdef __CUDA_ARCH__
    cuPrintf("%f ",f.weight) ;
    for ( int i = 0 ; i < dims ; i++ )
        cuPrintf("%f ",f.mean[i]) ;
    for ( int i = 0 ; i < dims*dims ; i++ )
        cuPrintf("%f ",f.cov[i]) ;
    cuPrintf("\n") ;
#else
    cout << f.weight << " " ;
    for ( int i = 0 ; i < dims ; i++ )
        cout << f.mean[i] << " " ;
    for ( int i = 0 ; i < dims*dims ; i++)
        cout << f.cov[i] << " " ;
    cout << endl ;
#endif
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

/// host-side log-sum-exponent for STL vector of doubles
double logSumExp( vector<double>log_terms )
{
    double val = log_terms[0] ;
    double sum = 0 ;
    for ( int i = 0 ; i < log_terms.size() ; i++ )
    {
        if ( log_terms[i] > val )
            val = log_terms[i] ;
    }
    for ( int i = 0 ; i < log_terms.size() ; i++ )
    {
        sum += exp(log_terms[i]-val) ;
    }
    sum = safeLog(sum) + val ;
    return sum ;
}

/// host-side log-sum-exponent for STL vector of floats
double logSumExp( vector<float>log_terms )
{
    double val = log_terms[0] ;
    double sum = 0 ;
    for ( int i = 0 ; i < log_terms.size() ; i++ )
    {
        if ( log_terms[i] > val )
            val = log_terms[i] ;
    }
    for ( int i = 0 ; i < log_terms.size() ; i++ )
    {
        sum += exp(log_terms[i]-val) ;
    }
    sum = safeLog(sum) + val ;
    return sum ;
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
    // set birth weight
    feature_birth.weight = safeLog(dev_config.birthWeight) ;

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
    feature_birth.weight = safeLog(dev_config.birthWeight) ;
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
        // partially update weight (log-transformed)
        features_update[idx].weight = safeLog(feature_pd)
                + safeLog(feature_predict.weight)
                - 0.5*dist
                - safeLog(2*M_PI)
                - 0.5*safeLog(det_sigma) ;
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
        // partially update weight (log-transformed)
        features_update[idx].weight = safeLog(feature_pd)
                + safeLog(feature_predict.weight)
                - 0.5*dist
                - safeLog(2*M_PI)
                - 0.5*safeLog(det_sigma) ;
    }
}


/// computes various components for the Kalman update of a particular feature
/*!
  * Given a vehicle pose and feature Gaussian, the function computes the Kalman
  * gain, updated covariance, innovation covariance, determinant of the
  * innovation covariance, probability of detection, and predicted measurement.
  * The computed values are stored at the addresses referenced by the passed
  * pointers.
  *
  * This code is specific to XY-heading vehicle state with range-bearing
  * measurements to XY point features.
  \param pose vehicle pose
  \param feature feature gaussian
  \param K pointer to store Kalman gain matrix
  \param cov_update pointer to store updated covariance matrix
  \param det_sigma pointer to store determinant of innov. covariance
  \param S pointer to store innov. covariance matrix.
  \param feature_pd pointer to store feature probability of detect.
  \param z_predict pointer to store predicted measurement
  */
__device__ void
computePreUpdateComponents( ConstantVelocityState pose,
                            Gaussian2D feature, REAL* K,
                            REAL* cov_update, REAL* det_sigma,
                            REAL* S, REAL* feature_pd,
                            RangeBearingMeasurement* z_predict )
{
    // predicted measurement
    REAL dx = feature.mean[0] - pose.px ;
    REAL dy = feature.mean[1] - pose.py ;
    REAL r2 = dx*dx + dy*dy ;
    REAL r = sqrt(r2) ;
    REAL bearing = wrapAngle(atan2f(dy,dx) - pose.ptheta) ;

    z_predict->range = r ;
    z_predict->bearing = bearing ;

    // probability of detection
    if ( r <= dev_config.maxRange && fabsf(bearing) <= dev_config.maxBearing )
        *feature_pd = dev_config.pd ;
    else
        *feature_pd = 0 ;

    // measurement jacobian wrt feature
    REAL J[4] ;
    J[0] = dx/r ;
    J[2] = dy/r ;
    J[1] = -dy/r2 ;
    J[3] = dx/r2 ;

    // predicted feature covariance
    REAL* P = feature.cov ;

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
//			makePositiveDefinite(sigma) ;

    *det_sigma = sigma[0]*sigma[3] - sigma[1]*sigma[2] ;

    S[0] = sigma[3]/(*det_sigma) ;
    S[1] = -sigma[1]/(*det_sigma) ;
    S[2] = -sigma[2]/(*det_sigma) ;
    S[3] = sigma[0]/(*det_sigma) ;

    // Kalman gain
    K[0] = S[0]*(P[0]*J[0] + P[2]*J[2]) + S[1]*(P[0]*J[1] + P[2]*J[3]) ;
    K[1] = S[0]*(P[1]*J[0] + P[3]*J[2]) + S[1]*(P[1]*J[1] + P[3]*J[3]) ;
    K[2] = S[2]*(P[0]*J[0] + P[2]*J[2]) + S[3]*(P[0]*J[1] + P[2]*J[3]) ;
    K[3] = S[2]*(P[1]*J[0] + P[3]*J[2]) + S[3]*(P[1]*J[1] + P[3]*J[3]) ;

    // Updated covariance (Joseph Form)
    cov_update[0] = ((1 - K[0] * J[0] - K[2] * J[1]) * P[0] + (-K[0] * J[2] - K[2] * J[3]) * P[1]) * (1 - K[0] * J[0] - K[2] * J[1]) + ((1 - K[0] * J[0] - K[2] * J[1]) * P[2] + (-K[0] * J[2] - K[2] * J[3]) * P[3]) * (-K[0] * J[2] - K[2] * J[3]) + pow(K[0], 2) * dev_config.stdRange*dev_config.stdRange + pow(K[2], 2) * dev_config.stdBearing*dev_config.stdBearing;
    cov_update[2] = ((1 - K[0] * J[0] - K[2] * J[1]) * P[0] + (-K[0] * J[2] - K[2] * J[3]) * P[1]) * (-K[1] * J[0] - K[3] * J[1]) + ((1 - K[0] * J[0] - K[2] * J[1]) * P[2] + (-K[0] * J[2] - K[2] * J[3]) * P[3]) * (1 - K[1] * J[2] - K[3] * J[3]) + K[0] * dev_config.stdRange*dev_config.stdRange * K[1] + K[2] * dev_config.stdBearing*dev_config.stdBearing * K[3];
    cov_update[1] = ((-K[1] * J[0] - K[3] * J[1]) * P[0] + (1 - K[1] * J[2] - K[3] * J[3]) * P[1]) * (1 - K[0] * J[0] - K[2] * J[1]) + ((-K[1] * J[0] - K[3] * J[1]) * P[2] + (1 - K[1] * J[2] - K[3] * J[3]) * P[3]) * (-K[0] * J[2] - K[2] * J[3]) + K[0] * dev_config.stdRange*dev_config.stdRange * K[1] + K[2] * dev_config.stdBearing*dev_config.stdBearing * K[3];
    cov_update[3] = ((-K[1] * J[0] - K[3] * J[1]) * P[0] + (1 - K[1] * J[2] - K[3] * J[3]) * P[1]) * (-K[1] * J[0] - K[3] * J[1]) + ((-K[1] * J[0] - K[3] * J[1]) * P[2] + (1 - K[1] * J[2] - K[3] * J[3]) * P[3]) * (1 - K[1] * J[2] - K[3] * J[3]) + pow(K[1], 2) * dev_config.stdRange*dev_config.stdRange + pow(K[3], 2) * dev_config.stdBearing*dev_config.stdBearing;
}

__device__ void
computePreUpdateComponentsDynamic( ConstantVelocityState pose,
                            Gaussian4D feature, REAL* K,
                            REAL* cov_update, REAL* det_sigma,
                            REAL* S, REAL* feature_pd,
                            RangeBearingMeasurement* z_predict )
{
    // predicted measurement
    REAL dx = feature.mean[0] - pose.px ;
    REAL dy = feature.mean[1] - pose.py ;
    REAL r2 = dx*dx + dy*dy ;
    REAL r = sqrt(r2) ;
    REAL bearing = wrapAngle(atan2f(dy,dx) - pose.ptheta) ;

    z_predict->range = r ;
    z_predict->bearing = bearing ;

    // probability of detection
    if ( r <= dev_config.maxRange && fabsf(bearing) <= dev_config.maxBearing )
        *feature_pd = dev_config.pd ;
    else
        *feature_pd = 0 ;

    // measurement jacobian wrt feature
    REAL J[4] ;
    J[0] = dx/r ;
    J[2] = dy/r ;
    J[1] = -dy/r2 ;
    J[3] = dx/r2 ;

    // predicted feature covariance
    REAL* P = feature.cov ;

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

    *det_sigma = sigma[0]*sigma[3] - sigma[1]*sigma[2] ;

    S[0] = sigma[3]/(*det_sigma) ;
    S[1] = -sigma[1]/(*det_sigma) ;
    S[2] = -sigma[2]/(*det_sigma) ;
    S[3] = sigma[0]/(*det_sigma) ;

    // Kalman gain
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
}

/// kernel for computing various constants used in the CPHD filter
__global__ void
cphdConstantsKernel( REAL* dev_factorial, REAL* dev_C, REAL* dev_cn_clutter )
{
    int n = threadIdx.x ;
    int k = blockIdx.x ;
    REAL* factorial = (REAL*)shmem ;

    factorial[n] = dev_factorial[n] ;
    __syncthreads() ;

    // compute the log binomial coefficients (nchoosek)
    int stride = dev_config.maxCardinality + 1 ;
    int idx = k*stride + n ;
    REAL log_nchoosek = 0 ;
    if ( k == 0 )
    {
        log_nchoosek = 0 ;
    }
    else if ( n == 0 || k > n )
    {
        log_nchoosek = LOG0 ;
    }
    else
    {
        log_nchoosek = factorial[n] - factorial[k]
                - factorial[n-k] ;
    }
    dev_C[idx] = log_nchoosek ;


    // thread block 0 computes the clutter cardinality
    if ( k == 0 )
    {
        dev_cn_clutter[n] = n*safeLog(dev_config.clutterRate)
                - dev_config.clutterRate
                - factorial[n] ;
    }
//	// for debugging: clutter cardinality with constant number of clutter
//	if ( k== 0 )
//	{
//		if ( n == dev_config.clutterRate)
//			dev_cn_clutter[n] = 0 ;
//		else
//			dev_cn_clutter[n] = LOG0 ;
//	}

}

/// host-side helper function to call cphdConstantsKernel
void
initCphdConstants()
{
    log_factorials = (REAL*)malloc( (config.maxCardinality+1)*sizeof(REAL) ) ;
    log_factorials[0] = 0 ;
    for ( int n = 1 ; n <= config.maxCardinality ; n++ )
    {
        log_factorials[n] = log_factorials[n-1] + safeLog((REAL)n) ;
    }
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_C,
                                pow(config.maxCardinality+1,2)*sizeof(REAL) ) ) ;
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_factorial,
                                (config.maxCardinality+1)*sizeof(REAL) ) ) ;
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_cn_clutter,
                                (config.maxCardinality+1)*sizeof(REAL) ) ) ;
    CUDA_SAFE_CALL( cudaMemcpy( dev_factorial, &log_factorials[0],
                                (config.maxCardinality+1)*sizeof(REAL),
                                cudaMemcpyHostToDevice ) ) ;
    CUDA_SAFE_THREAD_SYNC() ;
//	CUDA_SAFE_CALL(
//				cudaMalloc( (void**)&dev_pspower,
//							(config.maxCardinality+1)*sizeof(REAL) ) ) ;
//	CUDA_SAFE_CALL(
//				cudaMalloc( (void**)&dev_qspower,
//							(config.maxCardinality+1)*sizeof(REAL) ) ) ;

    int n_blocks = config.maxCardinality+1 ;
    int n_threads = n_blocks ;
    cphdConstantsKernel<<<n_blocks, n_threads, n_threads*sizeof(REAL)>>>
        ( dev_factorial, dev_C, dev_cn_clutter ) ;
    CUDA_SAFE_THREAD_SYNC() ;
}

/// kernel for particle prediction with an ackerman steering motion model
__global__ void
phdPredictKernelAckerman(ConstantVelocityState* particles_prior,
                   AckermanControl control,
                   AckermanNoise* noise,
                   ConstantVelocityState* particles_predict)
{
    const int tid = threadIdx.x ;
    const int predict_idx = blockIdx.x*blockDim.x + tid ;
    if (predict_idx < dev_config.n_particles*dev_config.nPredictParticles)
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
        ConstantVelocityNoise* noise, ConstantVelocityState* particles_predict )
{
    const int tid = threadIdx.x ;
    const int predict_idx = blockIdx.x*blockDim.x + tid ;
    if (predict_idx < dev_config.n_particles*dev_config.nPredictParticles)
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
            REAL p_jmm = 1/(1+exp(dev_config.beta*(dev_config.tau - v_mag))) ;
            features_predict[idx] = model.compute_prediction(features_prior[idx],
                                                             dev_config.dt) ;
            features_predict[idx].weight = p_jmm
                    *dev_config.ps*features_predict[idx].weight ;

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
predictMapMixed(ParticleSLAM& particles)
{
    // combine all dynamic features into one vector
    vector<Gaussian4D> all_features = combineFeatures(particles.maps_dynamic) ;
    int n_features = all_features.size() ;

    // allocate memory
    Gaussian4D* dev_features_prior = NULL ;
    Gaussian4D* dev_features_predict = NULL ;
    Gaussian2D* dev_features_jump = NULL ;
    CUDA_SAFE_CALL(cudaMalloc((void**)&dev_features_prior,
                              n_features*sizeof(Gaussian4D) ) ) ;
    CUDA_SAFE_CALL(cudaMalloc((void**)&dev_features_predict,
                              n_features*sizeof(Gaussian4D) ) ) ;
    CUDA_SAFE_CALL(cudaMalloc((void**)&dev_features_jump,
                              n_features*sizeof(Gaussian2D) ) ) ;
    CUDA_SAFE_CALL(cudaMemcpy(dev_features_prior,&all_features[0],
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
    CUDA_SAFE_CALL(cudaMemcpy(&all_features[0],dev_features_predict,
                              n_features*sizeof(Gaussian4D),
                              cudaMemcpyDeviceToHost)) ;
    CUDA_SAFE_CALL(cudaMemcpy(&all_features_jump[0],dev_features_jump,
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
        particles.maps_static[n].insert(particles.maps_static[n].end(),
                                        begin_jump,
                                        end_jump ) ;
        if ( n < particles.n_particles - 1)
        {
            begin = end ;
            end += particles.maps_dynamic[n+1].size() ;

            begin_jump = end_jump ;
            end_jump += particles.maps_dynamic[n+1].size() ;
        }
    }

    cout << "first predicted static feature" << endl ;
    Gaussian2D feature_test = particles.maps_static[0][0] ;
    cout << feature_test.weight << " "
         << feature_test.mean[0] << " " << feature_test.mean[1] << " "
         << feature_test.cov[0] << " " << feature_test.cov[1] << " "
         << feature_test.cov[2] << " " << feature_test.cov[3] << endl ;

    cout << "first predicted dynamic feature: " << endl ;
    print_feature(particles.maps_dynamic[0][0]) ;

    // free memory
    CUDA_SAFE_CALL( cudaFree( dev_features_prior ) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_features_predict ) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_features_jump ) ) ;
}

//template <class GaussianType>
//void
//predictMap(ParticleSLAM& particles)
//{
//    // combine all dynamic features into one vector
//    vector<Gaussian4D> all_features = combineFeatures(particles.maps_dynamic) ;
//    int n_features = all_features.size() ;
//    GaussianType* dev_features_prior = NULL ;
//    GaussianType* dev_features_predict = NULL ;
//    CUDA_SAFE_CALL(cudaMalloc((void**)&dev_features_prior,
//                              n_features*sizeof(Gaussian4D) ) ) ;
//    CUDA_SAFE_CALL(cudaMalloc((void**)&dev_features_predict,
//                              n_features*sizeof(Gaussian4D) ) ) ;
//    CUDA_SAFE_CALL(cudaMemcpy(dev_features_prior,&all_features[0],
//                              n_features*sizeof(Gaussian4D),
//                              cudaMemcpyHostToDevice) ) ;
//    int n_blocks = (n_features+255)/256 ;
//    ConstantVelocityMotionModel motion_model ;
//    motion_model.std_accx = config.stdAxMap ;
//    motion_model.std_accy = config.stdAyMap ;
//    predictMapKernel<<<n_blocks,256>>>
//        (dev_features_prior,motion_model,n_features, dev_features_predict ) ;
//    CUDA_SAFE_CALL(cudaMemcpy(&all_features[0],dev_features_predict,
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
//    CUDA_SAFE_CALL( cudaFree( dev_features_prior ) ) ;
//    CUDA_SAFE_CALL( cudaFree( dev_features_predict ) ) ;
//}

/// host-side helper function for PHD filter prediction
void
phdPredict(ParticleSLAM& particles, ... )
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
    CUDA_SAFE_CALL(
                cudaMalloc((void**)&dev_states_prior,
                           n_particles*sizeof(ConstantVelocityState) ) ) ;
    CUDA_SAFE_CALL(
                cudaMalloc((void**)&dev_states_predict,
                           nPredict*sizeof(ConstantVelocityState) ) ) ;

    // copy inputs
    CUDA_SAFE_CALL(
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
            noiseVector[i].atheta = 3*config.atheta * randn() ;
        }

        ConstantVelocityNoise* dev_noise = NULL ;
        CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_noise,
                               n_particles*sizeof(ConstantVelocityNoise) ) ) ;
        CUDA_SAFE_CALL(
                    cudaMemcpy(dev_noise, &noiseVector[0],
                               n_particles*sizeof(ConstantVelocityNoise),
                               cudaMemcpyHostToDevice) ) ;

        // launch the kernel
        int nThreads = min(nPredict,256) ;
        int nBlocks = (nPredict+255)/256 ;
        phdPredictKernel
        <<<nBlocks, nThreads>>>
        ( dev_states_prior,dev_noise,dev_states_predict ) ;

        cudaFree(dev_noise) ;
    }
    else if( config.motionType == ACKERMAN_MOTION )
    {
        // read in the control data structure from variable argument lest
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
        CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_noise,
                               nPredict*sizeof(AckermanNoise) ) ) ;
        CUDA_SAFE_CALL(
                    cudaMemcpy(dev_noise, &noiseVector[0],
                               nPredict*sizeof(AckermanNoise),
                               cudaMemcpyHostToDevice) ) ;

        // launch the kernel
        int nThreads = min(nPredict,256) ;
        int nBlocks = (nPredict+255)/256 ;
        phdPredictKernelAckerman
        <<<nBlocks, nThreads>>>
        (dev_states_prior,control,dev_noise,dev_states_predict) ;

        cudaFree(dev_noise) ;
    }


    // copy results from device
    ConstantVelocityState* states_predict = (ConstantVelocityState*)
                                            malloc(nPredict*sizeof(ConstantVelocityState)) ;
    CUDA_SAFE_CALL(cudaMemcpy(states_predict, dev_states_predict,
                              nPredict*sizeof(ConstantVelocityState),
                              cudaMemcpyDeviceToHost) ) ;
    particles.states.assign( states_predict, states_predict+nPredict ) ;



    // duplicate the PHD filter maps and cardinalities for the newly spawned
    // vehicle particles, and downscale particle weights
    if ( config.nPredictParticles > 1 )
    {
        vector<vector<Gaussian2D> > maps_predict_static ;
        vector<vector<Gaussian4D> > maps_predict_dynamic ;
        vector<double> weights_predict ;
        vector< vector <REAL> > cardinalities_predict ;
        maps_predict_static.clear();
        maps_predict_static.reserve(nPredict);
        maps_predict_dynamic.clear();
        maps_predict_dynamic.reserve(nPredict);
        weights_predict.clear();
        weights_predict.reserve(nPredict);
        cardinalities_predict.clear();
        cardinalities_predict.reserve(nPredict);
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
            weights_predict.insert( weights_predict.end(), config.nPredictParticles,
                                    particles.weights[i] - safeLog(config.nPredictParticles) ) ;
        }
//        DEBUG_VAL(maps_predict.size()) ;
        particles.maps_static = maps_predict_static ;
        particles.maps_dynamic = maps_predict_dynamic ;
        particles.weights = weights_predict ;
        particles.cardinalities = cardinalities_predict ;
        particles.n_particles = nPredict ;
    }

    // map prediction
    if(config.dynamicFeatures)
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
    CUDA_SAFE_CALL( cudaFree( dev_states_prior ) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_states_predict ) ) ;
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

/// generates a binomial Poisson cardinality distribution for the in-range features.
__global__ void
separateCardinalityKernel( Gaussian2D *features, int* map_offsets,
                           REAL* cn_inrange)
{
    int n = threadIdx.x ;
    int map_idx = blockIdx.x ;
    int n_features = map_offsets[map_idx+1] - map_offsets[map_idx] ;
    int feature_idx = map_offsets[map_idx] + n ;
    REAL* cn_shared = (REAL*)shmem ;
    REAL* weights = (REAL*)&cn_shared[dev_config.maxCardinality+1] ;

    // compute product of weights
    REAL val = 0 ;
    if ( n < n_features )
    {
        val = log(features[ feature_idx ].weight) ;
    }
    sumByReduction( weights, val, n ) ;
    REAL log_alpha = weights[0] ;
    __syncthreads() ;

    // load the polynomial roots into shared memory
    if ( n < n_features )
    {
        weights[n] = (1-features[feature_idx].weight)/features[feature_idx].weight ;
    }
    else
    {
        weights[n] = 0 ;
    }

    // compute full cn using recursive algorithm
    cn_shared[n+1] = 0 ;
    int cn_offset = map_idx*(dev_config.maxCardinality+1) ;
    if ( n == 0 )
    {
        cn_shared[0] = 1 ;
    }
    __syncthreads() ;
    for ( int m = 0 ; m < n_features ; m++ )
    {
        REAL tmp1 = cn_shared[n+1] ;
        REAL tmp2 = cn_shared[n] ;
        __syncthreads() ;
        if ( n < m+1 )
            cn_shared[n+1] = tmp1 - weights[m]*tmp2 ;
        __syncthreads() ;
    }
    if ( n <= n_features )
    {
        int idx = cn_offset + (n_features - n) ;
        cn_inrange[idx] = safeLog(fabs(cn_shared[n]))
                + log_alpha ;
    }
    else
    {
        cn_inrange[cn_offset+n] = LOG0 ;
    }
}

/// compute partially updated weights and updated means & covariances
/**
  \param features Array of all Gaussians from all particles concatenated together
  \param map_sizes Integer array indicating the number of features per particle.
  \param n_particles Number of particles
  \param n_measurements Number of measurements
  \param poses Array of particle poses
  \param w_partial Array of partially updated weights computed by kernel
  */
__global__ void
cphdPreUpdateKernel(Gaussian2D *features, int* map_offsets,
        int n_particles, int n_measurements, ConstantVelocityState* poses,
        Gaussian2D* updated_features, REAL* w_partial, REAL* qdw )

{
    int tid = threadIdx.x + blockIdx.x*blockDim.x ;
    int n_total = (n_measurements+1)*map_offsets[n_particles] ;
    if ( tid >= n_total)
        return ;
    int map_idx = 0 ;
    while ( map_offsets[map_idx]*(n_measurements+1) <= tid )
    {
        map_idx++ ;
    }
    map_idx-- ;
    int n_features = map_offsets[map_idx+1] - map_offsets[map_idx] ;
    int offset = map_offsets[map_idx]*(n_measurements+1) ;
    int feature_idx = floor( (float)(tid-offset)/(n_measurements) ) ;

    if ( feature_idx >= n_features ) // non-detect thread
    {
        int predict_idx = tid - n_features*n_measurements - offset
                + map_offsets[map_idx] ;
        updated_features[tid] = features[predict_idx] ;
    }
    else if ( tid < n_total ) // update thread
    {
        int z_idx = tid - feature_idx*n_measurements - offset ;

        Gaussian2D feature = features[map_offsets[map_idx]+feature_idx] ;
        Gaussian2D updated_feature ;
        RangeBearingMeasurement z = Z[z_idx] ;
        RangeBearingMeasurement z_predict ;
        ConstantVelocityState pose = poses[map_idx] ;
        REAL K[4] = {0,0,0,0} ;
        REAL sigmaInv[4] = {0,0,0,0} ;
        REAL covUpdate[4] = {0,0,0,0} ;
        REAL featurePd = 0 ;
        REAL detSigma = 0 ;

//        // predicted measurement
//        REAL dx = feature.mean[0] - pose.px ;
//        REAL dy = feature.mean[1] - pose.py ;
//        REAL r2 = dx*dx + dy*dy ;
//        REAL r = sqrt(r2) ;
//        REAL bearing = wrapAngle(atan2f(dy,dx) - pose.ptheta) ;
//        REAL featurePd = 0 ;

//        // probability of detection
//        if ( r <= dev_config.maxRange && fabsf(bearing) <= dev_config.maxBearing )
//            featurePd = dev_config.pd ;

//        // declare matrices
//        REAL J[4] = {0,0,0,0} ;
//        REAL sigma[4] = {0,0,0,0} ;
//        REAL sigmaInv[4] = {0,0,0,0} ;
//        REAL K[4] = {0,0,0,0} ;
//        REAL detSigma = 0 ;

//        // measurement jacobian wrt feature
//        J[0] = dx/r ;
//        J[2] = dy/r ;
//        J[1] = -dy/r2 ;
//        J[3] = dx/r2 ;

//        // BEGIN Maple-Generated expressions
//    #define P feature.cov
//    #define S sigmaInv
//        // innovation covariance
//        sigma[0] = (P[0] * J[0] + J[2] * P[1]) * J[0] + (J[0] * P[2] + P[3] * J[2]) * J[2] + pow(dev_config.stdRange,2) ;
//        sigma[1] = (P[0] * J[1] + J[3] * P[1]) * J[0] + (J[1] * P[2] + P[3] * J[3]) * J[2];
//        sigma[2] = (P[0] * J[0] + J[2] * P[1]) * J[1] + (J[0] * P[2] + P[3] * J[2]) * J[3];
//        sigma[3] = (P[0] * J[1] + J[3] * P[1]) * J[1] + (J[1] * P[2] + P[3] * J[3]) * J[3] + pow(dev_config.stdBearing,2) ;

//        // enforce symmetry
//        sigma[1] = (sigma[1]+sigma[2])/2 ;
//        sigma[2] = sigma[1] ;
//    //			makePositiveDefinite(sigma) ;

//        detSigma = sigma[0]*sigma[3] - sigma[1]*sigma[2] ;
//        sigmaInv[0] = sigma[3]/detSigma ;
//        sigmaInv[1] = -sigma[1]/detSigma ;
//        sigmaInv[2] = -sigma[2]/detSigma ;
//        sigmaInv[3] = sigma[0]/detSigma ;

//        // Kalman gain
//        K[0] = S[0]*(P[0]*J[0] + P[2]*J[2]) + S[1]*(P[0]*J[1] + P[2]*J[3]) ;
//        K[1] = S[0]*(P[1]*J[0] + P[3]*J[2]) + S[1]*(P[1]*J[1] + P[3]*J[3]) ;
//        K[2] = S[2]*(P[0]*J[0] + P[2]*J[2]) + S[3]*(P[0]*J[1] + P[2]*J[3]) ;
//        K[3] = S[2]*(P[1]*J[0] + P[3]*J[2]) + S[3]*(P[1]*J[1] + P[3]*J[3]) ;

//        // Updated covariance (Joseph Form)
//        updated_feature.cov[0] = ((1 - K[0] * J[0] - K[2] * J[1]) * P[0] + (-K[0] * J[2] - K[2] * J[3]) * P[1]) * (1 - K[0] * J[0] - K[2] * J[1]) + ((1 - K[0] * J[0] - K[2] * J[1]) * P[2] + (-K[0] * J[2] - K[2] * J[3]) * P[3]) * (-K[0] * J[2] - K[2] * J[3]) + pow(K[0], 2) * dev_config.stdRange*dev_config.stdRange + pow(K[2], 2) * dev_config.stdBearing*dev_config.stdBearing;
//        updated_feature.cov[2] = ((1 - K[0] * J[0] - K[2] * J[1]) * P[0] + (-K[0] * J[2] - K[2] * J[3]) * P[1]) * (-K[1] * J[0] - K[3] * J[1]) + ((1 - K[0] * J[0] - K[2] * J[1]) * P[2] + (-K[0] * J[2] - K[2] * J[3]) * P[3]) * (1 - K[1] * J[2] - K[3] * J[3]) + K[0] * dev_config.stdRange*dev_config.stdRange * K[1] + K[2] * dev_config.stdBearing*dev_config.stdBearing * K[3];
//        updated_feature.cov[1] = ((-K[1] * J[0] - K[3] * J[1]) * P[0] + (1 - K[1] * J[2] - K[3] * J[3]) * P[1]) * (1 - K[0] * J[0] - K[2] * J[1]) + ((-K[1] * J[0] - K[3] * J[1]) * P[2] + (1 - K[1] * J[2] - K[3] * J[3]) * P[3]) * (-K[0] * J[2] - K[2] * J[3]) + K[0] * dev_config.stdRange*dev_config.stdRange * K[1] + K[2] * dev_config.stdBearing*dev_config.stdBearing * K[3];
//        updated_feature.cov[3] = ((-K[1] * J[0] - K[3] * J[1]) * P[0] + (1 - K[1] * J[2] - K[3] * J[3]) * P[1]) * (-K[1] * J[0] - K[3] * J[1]) + ((-K[1] * J[0] - K[3] * J[1]) * P[2] + (1 - K[1] * J[2] - K[3] * J[3]) * P[3]) * (1 - K[1] * J[2] - K[3] * J[3]) + pow(K[1], 2) * dev_config.stdRange*dev_config.stdRange + pow(K[3], 2) * dev_config.stdBearing*dev_config.stdBearing;
//    #undef P
//    #undef S

        computePreUpdateComponents( pose, feature, K, covUpdate,
                                    &detSigma, sigmaInv, &featurePd,
                                    &z_predict ) ;

        // innovation
        REAL innov[2] = {0,0} ;
        innov[0] = z.range - z_predict.range ;
        innov[1] = wrapAngle(z.bearing - z_predict.bearing) ;

        // updated mean
        updated_feature.mean[0] = feature.mean[0] + K[0]*innov[0] + K[2]*innov[1] ;
        updated_feature.mean[1] = feature.mean[1] + K[1]*innov[0] + K[3]*innov[1] ;

        // updated covariances
        updated_feature.cov[0] = covUpdate[0] ;
        updated_feature.cov[1] = covUpdate[1] ;
        updated_feature.cov[2] = covUpdate[2] ;
        updated_feature.cov[3] = covUpdate[3] ;

        // single-object likelihood
        REAL dist = innov[0]*innov[0]*sigmaInv[0] +
                innov[0]*innov[1]*(sigmaInv[1] + sigmaInv[2]) +
                innov[1]*innov[1]*sigmaInv[3] ;

        // partially updated weight
        updated_feature.weight = safeLog(featurePd) + safeLog(feature.weight)
                - 0.5*dist- safeLog(2*M_PI) - 0.5*safeLog(detSigma) ;

        updated_features[tid] = updated_feature ;

        int w_idx = map_offsets[map_idx]*n_measurements ;
        w_idx += feature_idx*n_measurements + z_idx ;
        w_partial[w_idx] = updated_feature.weight ;

        if ( z_idx == 0 )
        {
            offset = map_offsets[map_idx] ;
            qdw[offset+feature_idx] = safeLog(1-featurePd) + safeLog(feature.weight) ;
        }
    }
}

/// computes the elementary symmetric polynomial coefficients
/**
  This kernel produces the coefficients of the elementary symmetric function
  for the CPHD update

  \param w_partial Array of partially updated weights
  \param map_sizes Number of features per particle
  \param n_measurements Number of measurements
  \param esf Array of ESF coefficients computed by kernel
  \param esfd Array of ESF coefficients, with each measurement omitted
  */
__global__ void
computeEsfKernel( REAL* w_partial, int* map_offsets, int n_measurements,
                  REAL* esf, REAL* esfd )
{
    REAL* lambda = (REAL*)shmem ;
    REAL* esf_shared = (REAL*)&lambda[n_measurements] ;

    // determine indexing offsets
    int tid = threadIdx.x ;
    int map_idx = blockIdx.x ;
    int block_offset = n_measurements*map_offsets[map_idx] ;
    int n_features = map_offsets[map_idx+1] - map_offsets[map_idx] ;

    // compute log lambda
    lambda[tid] = 0 ;
    int idx = block_offset + tid ;
    REAL max_val = -FLT_MAX ;
    for ( int j = 0 ; j < n_features ; j++)
    {
        REAL tmp = w_partial[idx] ;
        REAL tmp_max = fmax(tmp,max_val) ;
        lambda[tid] = exp( max_val - tmp_max )*lambda[tid]
                + exp( tmp - tmp_max ) ;
        max_val = tmp_max ;
        idx += n_measurements ;
    }
    lambda[tid] = safeLog(lambda[tid]) + max_val
            + safeLog(dev_config.clutterRate)
            - safeLog(dev_config.clutterDensity) ;
    __syncthreads() ;

    // compute full esf using recursive algorithm
    esf_shared[tid+1] = 0 ;
    int esf_offset = map_idx*(n_measurements+1) ;
    if ( tid == 0 )
    {
        esf_shared[0] = 1 ;
        esf[esf_offset] = 0 ;
    }
    __syncthreads() ;
    for ( int m = 0 ; m < n_measurements ; m++ )
    {
        REAL tmp1 = esf_shared[tid+1] ;
        REAL tmp2 = esf_shared[tid] ;
        __syncthreads() ;
        if ( tid < m+1 )
        {
//            REAL tmp_sum ;
//            max_val = fmax(tmp1, lambda[m]+tmp2) ;
//            tmp_sum = exp(tmp1-max_val) + exp(lambda[m]+tmp2-max_val) ;
//            esf_shared[tid+1] = safeLog( fabs(tmp_sum) ) + max_val ;
            esf_shared[tid+1] = tmp1 - exp(lambda[m])*tmp2 ;
        }
        __syncthreads() ;
    }
    esf[esf_offset+tid+1] = log(fabs(esf_shared[tid+1])) ;

    // compute esf's for detection terms
    for ( int m = 0 ; m < n_measurements ; m++ )
    {
        int esfd_offset = n_measurements*n_measurements*map_idx + m*n_measurements ;
//        esf_shared[tid+1] = LOG0 ;
        esf_shared[tid+1] = 0 ;
        if ( tid == 0 )
        {
//            esf_shared[0] = 0 ;
//            esfd[esfd_offset] = 0 ;
            esf_shared[0] = 1 ;
            esfd[esfd_offset] = 0 ;
        }
        __syncthreads() ;
        int k = 0 ;
        for ( int n = 0 ; n < n_measurements ; n++ )
        {
            REAL tmp1 = esf_shared[tid+1] ;
            REAL tmp2 = esf_shared[tid] ;
            __syncthreads() ;
            if ( n != m )
            {
                if ( tid < k+1 )
                {
//                    REAL tmp_sum ;
//                    max_val = fmax(tmp1,lambda[n]+tmp2) ;
//                    tmp_sum = exp(tmp1-max_val) - exp(lambda[n]+tmp2-max_val) ;
//                    esf_shared[tid+1] = safeLog( fabs(tmp_sum) ) + max_val ;
                    esf_shared[tid+1] = tmp1 - exp(lambda[n])*tmp2 ;
                }
                k++ ;
            }
            __syncthreads() ;
        }
        if ( tid < (n_measurements-1) )
            esfd[esfd_offset+tid+1] = log(fabs(esf_shared[tid+1])) ;
    }
}

/// compute the multi-object likelihoods for the CPHD update
/**
  This kernel computes the terms denoted as Psi in Vo's Analytic CPHD paper, and
  their inner products with the predicted cardinality distribution. It also
  produces the updated cardinality
  */
__global__ void
computePsiKernel( Gaussian2D* features_predict, REAL* cn_predict, REAL* esf,
                  REAL* esfd, int* map_offsets,
                  int n_measurements, REAL* qdw, REAL* dev_factorial,
                  REAL* dev_C, REAL* dev_cn_clutter, REAL* cn_update,
                  REAL* innerprod_psi0, REAL* innerprod_psi1,
                  REAL* innerprod_psi1d )
{
    int n = threadIdx.x ;
    REAL psi0 = 0 ;
    REAL psi1 = 0 ;
    int map_idx = blockIdx.x ;
    int cn_offset = (dev_config.maxCardinality+1)*map_idx ;
    int esf_offset = (n_measurements+1)*map_idx ;
    int stop_idx = 0 ;
    REAL max_val0 = 0 ;
    REAL max_val1 = 0 ;
    REAL* shdata = (REAL*)shmem ;

    // compute the (log) inner product < q_D, w >
    int map_offset = map_offsets[map_idx] ;
    int n_features = map_offsets[map_idx+1] - map_offsets[map_idx] ;
    REAL innerprod_qdw = 0 ;
    max_val0 = qdw[map_offset] ;
    for ( int j = 0 ; j < n_features ; j+=blockDim.x )
    {
        REAL val = -FLT_MAX ;
        if ( j+n < n_features )
            val = qdw[map_offset+j+n] ;
        maxByReduction(shdata,val,n) ;
        max_val0 = fmax(max_val0,shdata[0]) ;
        __syncthreads() ;
    }
    for ( int j = 0 ; j < n_features ; j+= blockDim.x )
    {
        REAL val = 0 ;
        if ( (j+n) < n_features )
            val = exp(qdw[map_offset+j+n]-max_val0) ;
        sumByReduction( shdata, val, n ) ;
        innerprod_qdw += shdata[0] ;
        __syncthreads() ;
    }
    innerprod_qdw = safeLog(innerprod_qdw) + max_val0 ;

    // compute the (log) inner product < 1, w >
    REAL wsum = 0 ;
    for ( int j = 0 ; j < n_features ; j += blockDim.x )
    {
        REAL val = 0 ;
        if ( (j+n) < n_features )
            val = features_predict[map_offset+j+n].weight ;
        sumByReduction( shdata, val, n );
        wsum += shdata[0] ;
        __syncthreads() ;
    }
    wsum = safeLog(wsum) ;

    // compute (log) PSI0(n) and PSI1(n), using log-sum-exp
    max_val0 = -FLT_MAX ;
    max_val1 = -FLT_MAX ;
    stop_idx = min(n,n_measurements) ;
    for ( int j = 0 ; j <= stop_idx ; j++ )
    {
        // PSI0
        REAL p_coeff = dev_C[n+j*(dev_config.maxCardinality+1)]
                + dev_factorial[j] ;
        REAL aux = dev_factorial[n_measurements-j]
                + dev_cn_clutter[n_measurements-j] + esf[esf_offset+ j]
                - n*wsum ;
        REAL tmp =  aux + p_coeff + (n-j)*innerprod_qdw ;

        psi0 = exp(max_val0-fmax(max_val0,tmp))*psi0
                + exp(tmp - fmax(max_val0,tmp) ) ;
        max_val0 = fmax(max_val0,tmp) ;

        // PSI1
        p_coeff = dev_C[n+(j+1)*(dev_config.maxCardinality+1)]
                + dev_factorial[j+1] ;
        tmp = aux + p_coeff + (n-(j+1))*innerprod_qdw  ;
        psi1 = exp(max_val1-fmax(max_val1,tmp))*psi1
                + exp(tmp - fmax(max_val1,tmp) ) ;
        max_val1 = fmax(max_val1,tmp) ;
    }
    psi0 = safeLog(psi0) + max_val0 ;
    psi1 = safeLog(psi1) + max_val1 ;

    // (log) inner product of PSI0 and predicted cardinality distribution, using
    // log-sum-exp trick
    REAL val = psi0 + cn_predict[cn_offset+n] ;
    maxByReduction( shdata, val, n ) ;
    max_val0 = shdata[0] ;
    __syncthreads() ;
    sumByReduction( shdata, exp(val-max_val0), n ) ;
    if ( n==0 )
        innerprod_psi0[map_idx] = safeLog(shdata[0]) + max_val0 ;


    // (log) inner product of PSI1 and predicted cardinality distribution, using
    // log-sum-exp trick
    val = psi1 + cn_predict[cn_offset+n] ;
    maxByReduction( shdata, psi1+cn_predict[cn_offset+n], n ) ;
//	shdata[n] = psi1+cn_predict[cn_offset+n] ;
    max_val1 = shdata[0] ;
    __syncthreads() ;
    sumByReduction( shdata, exp( val - max_val1 ), n ) ;
    if ( n == 0 )
        innerprod_psi1[map_idx] = safeLog(shdata[0]) + max_val1 ;
//	__syncthreads() ;

    // PSI1 detection terms
    stop_idx = min(n_measurements - 1, n) ;
    for ( int m = 0 ; m < n_measurements ; m++ )
    {
        int esfd_offset = map_idx * n_measurements * n_measurements
                + m*n_measurements ;
        REAL psi1d = 0 ;
        max_val1 = -FLT_MAX ;
        for ( int j = 0 ; j <= stop_idx ; j++ )
        {
            REAL p_coeff = dev_C[n+(j+1)*(dev_config.maxCardinality+1)]
                    + dev_factorial[j+1] ;
            REAL aux = dev_factorial[n_measurements-1-j]
                + dev_cn_clutter[n_measurements-1-j] + esfd[esfd_offset+ j]
                - n*wsum ;
            REAL tmp = aux + p_coeff + (n-(j+1))*innerprod_qdw ;
            psi1d = exp(max_val1-fmax(max_val1,tmp))*psi1d
                    + exp(tmp - fmax(max_val1,tmp) ) ;
            max_val1 = fmax(max_val1,tmp) ;
        }
        psi1d = safeLog(psi1d) + max_val1 ;
        val = psi1d + cn_predict[cn_offset+n] ;
        maxByReduction( shdata, val, n ) ;
        max_val1 = shdata[0] ;
        __syncthreads() ;
        sumByReduction( shdata, exp(val-max_val1), n ) ;
        if ( n == 0 )
            innerprod_psi1d[map_idx*n_measurements+m] = safeLog(shdata[0]) + max_val1 ;
        __syncthreads() ;
    }

    // compute log updated cardinality
    cn_update[cn_offset+n] = cn_predict[cn_offset+n] + psi0
            - innerprod_psi0[map_idx] ;
}

/// perform the gaussian mixture CPHD weight update
/**
  This kernel takes the results produced by the previous three kernels in the
  CPHD pipeline (PreUpdate, ComputeEsf, and ComputePsi) and applies them to
  update the weights of the Gaussian Mixture as in Vo's paper

  Kernel organization: One thread block per particle. Each thread updates all
  the features for one measurement.
  */
__global__ void
cphdUpdateKernel( int* map_offsets, int n_measurements,
                  REAL* innerprod_psi0, REAL* innerprod_psi1,
                  REAL* innerprod_psi1d, bool* merged_flags,
                  Gaussian2D* updated_features )
{
    int z_idx = threadIdx.x ;
    int map_idx = blockIdx.x ;
    int offset = (n_measurements+1)*map_offsets[map_idx] ;
    int n_features = map_offsets[map_idx+1] - map_offsets[map_idx] ;

    // detection update
    REAL psi1d = innerprod_psi1d[n_measurements*map_idx+z_idx] ;
    for ( int j = 0 ; j < n_features ; j++ )
    {
        REAL tmp = updated_features[offset+z_idx].weight
                + psi1d - innerprod_psi0[map_idx] + safeLog(dev_config.clutterRate)
                - safeLog(dev_config.clutterDensity) ;
        updated_features[offset+z_idx].weight = exp(tmp) ;
        if ( exp(tmp) >= dev_config.minFeatureWeight )
            merged_flags[offset + z_idx] = false ;
        else
            merged_flags[offset + z_idx] = true ;
        offset += n_measurements ;
    }

    // non-detection updates
    for ( int j = 0 ; j < n_features ; j += blockDim.x )
    {
        if ( j+z_idx < n_features )
        {
            int nondetect_idx = offset + j + z_idx ;
            REAL tmp = safeLog(updated_features[nondetect_idx].weight)
                    + innerprod_psi1[map_idx] - innerprod_psi0[map_idx]
                    + safeLog(1-dev_config.pd) ;
            updated_features[nondetect_idx].weight = exp(tmp) ;
            if ( exp(tmp) >= dev_config.minFeatureWeight )
                merged_flags[nondetect_idx] = false ;
            else
                merged_flags[nondetect_idx] = true ;
        }
    }
}

/// perform the gaussian mixture PHD update
/**
  PHD update algorithm as in Vo & Ma 2006. Gaussian mixture reduction (merging),
  as in the clustering algorithm by Salmond 1990.
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
__global__ void
phdUpdateKernelStatic(ConstantVelocityState* poses,
                      Gaussian2D* features_predict,
                      int* map_offsets,
                      int n_particles, int n_measure,
                      Gaussian2D* features_update,
                      bool* merge_flags,
                      REAL* particle_weights)
{
    // shared memory variables
    __shared__ REAL sdata[256] ;


    ConstantVelocityState pose ;
    REAL particle_weight = 0 ;
    REAL cardinality_predict = 0 ;
    int map_idx = 0 ;
    int update_offset = 0 ;
    int n_features = 0 ;
    int n_update = 0 ;
    int predict_offset = 0 ;


    // initialize variables
    int tid = threadIdx.x ;
    // pre-update variables
    REAL featurePd = 0 ;
    Gaussian2D feature ;

    // update variables
    REAL w_partial = 0 ;
    int updateIdx = 0 ;

    // loop over particles
    for ( int p = 0 ; p < n_particles ; p += gridDim.x )
    {
        if ( p + blockIdx.x < n_particles )
        {
            // initialize map-specific variables
            map_idx = p + blockIdx.x ;
            predict_offset = map_offsets[map_idx] ;
            update_offset = predict_offset*(n_measure+1) +
                    map_idx*n_measure ;
            n_features = map_offsets[map_idx+1] - map_offsets[map_idx] ;
            n_update = (n_features)*(n_measure+1) + n_measure ;
            pose = poses[map_idx] ;
            particle_weight = 0 ;
            cardinality_predict = 0.0 ;

            // loop over predicted features + newborn features
            for ( int j = 0 ; j < (n_features+n_measure) ; j += blockDim.x )
            {
                int feature_idx = j + tid ;
                w_partial = 0 ;
                if ( feature_idx < n_features )
                {
                    // persistent feature

                    // get the feature corresponding to the current thread
                    feature = features_predict[predict_offset+feature_idx] ;

                    Gaussian2D* ptr_nondetect = features_update+update_offset
                            + feature_idx ;
                    Gaussian2D* ptr_update = ptr_nondetect + n_features ;
                    computePreUpdate(pose,feature,n_features,n_measure,
                                     featurePd,*ptr_nondetect,
                                     ptr_update);
                    w_partial = featurePd*feature.weight ;
                }
                else if (feature_idx < n_features+n_measure)
                {
                    // newborn feature

                    // find measurement corresponding to current thread
                    int z_idx = feature_idx - n_features ;

                    Gaussian2D* ptr_birth = features_update +
                            update_offset + n_features
                            + n_measure*n_features + z_idx ;
                    computeBirth(pose,Z[z_idx],*ptr_birth);

                    // set Pd to 1 for newborn features
                    w_partial = 0 ;
                    // no non-detection term for newborn features
                    if(tid==0 && map_idx==0)
                    {
                        if ( feature_idx == n_features)
                        {
                            cuPrintf("before: %f\n",ptr_birth->weight) ;
                            ptr_birth->weight = safeLog(dev_config.birthWeight) ;
                            cuPrintf("after: %f\n",ptr_birth->weight) ;
                        }
                    }
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

            // compute the weight normalizers
            for ( int i = 0 ; i < n_measure ; i++ )
            {
                REAL log_normalizer = 0 ;
                REAL val = 0 ;
                REAL sum = 0 ;
                Gaussian2D* ptr_update = features_update
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
                    particle_weight += log_normalizer ;
                }
            }


            // Particle weighting
            __syncthreads() ;
            if ( tid == 0 )
            {
                particle_weight -= cardinality_predict ;
                particle_weights[map_idx] = particle_weight ;
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
    int map_idx = 0 ;
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

    REAL cardinality_predict = 0 ;
    REAL particle_weight = 0 ;

    // loop over particles
    for ( int p = 0 ; p < n_particles ; p += gridDim.x )
    {
        map_idx = p + blockIdx.x ;
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
                        computePreUpdate( pose, features_predict_static[feature_idx],
                                          n_features_static, n_measure, feature_pd,
                                          *ptr_nondetect, ptr_update ) ;
                        val = feature_pd
                                *features_predict_static[feature_idx].weight ;

//                        if(feature_idx == 0)
//                        {
//                            cuPrintf("non-detect term: \n") ;
//                            print_feature(*ptr_nondetect) ;
//                        }
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
                        computePreUpdate( pose, features_predict_dynamic[feature_idx_dynamic],
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
                    if ( feature_idx < n_features_static+n_features_dynamic )
                    {
                        bool is_static = feature_idx < n_features_static ;
                        if ( is_static )
                        {
//                            val = ptr_static[feature_idx].weight ;
                            val = exp(ptr_static[feature_idx].weight) ;
                        }
                        else
                        {
//                            val = ptr_dynamic[feature_idx-n_features_static].weight ;
                            val = exp(ptr_dynamic[feature_idx-n_features_static].weight) ;
                        }
                    }
                    sumByReduction(sdata,val,tid);
                    normalizer += sdata[0] ;
                }
                normalizer += dev_config.clutterDensity
                        + 2*dev_config.birthWeight ;
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
            if ( tid == 0)
                particle_weights[map_idx] = particle_weight ;
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

template <class GaussianType>
__global__ void
phdUpdateMergeKernel(GaussianType* updated_features,
                     GaussianType* mergedFeatures, int *mergedSizes,
                     bool *mergedFlags, int* map_sizes_static, int n_particles )
{
    __shared__ GaussianType maxFeature ;
//    __shared__ REAL wMerge ;
//    __shared__ REAL meanMerge[2] ;
//    __shared__ REAL covMerge[4] ;
    __shared__ GaussianType mergedFeature ;
    __shared__ REAL sdata[256] ;
    __shared__ int mergedSize ;
    __shared__ int update_offset ;
    __shared__ int n_update ;
    int tid = threadIdx.x ;
    REAL dist ;
//    REAL innov[2] ;
    GaussianType feature ;
    clearGaussian(feature) ;
    int dims = getGaussianDim(feature) ;

    // loop over particles
    for ( int p = 0 ; p < n_particles ; p += gridDim.x )
    {
        int map_idx = p + blockIdx.x ;
        if ( map_idx <= n_particles )
        {
            // initialize shared vars
            if ( tid == 0)
            {
                update_offset = 0 ;
                for ( int i = 0 ; i < map_idx ; i++ )
                {
                    update_offset += map_sizes_static[i] ;
                }
                n_update = map_sizes_static[map_idx] ;
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
    CUDA_SAFE_CALL(
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
        CUDA_SAFE_CALL(
                    cudaMalloc( (void**)&dev_maps,
                                total_features*sizeof(GaussianType) ) ) ;;
        CUDA_SAFE_CALL(
                    cudaMalloc( (void**)&dev_n_in_range,
                                n_particles*sizeof(int) ) ) ;
        CUDA_SAFE_CALL(
                    cudaMalloc( (void**)&dev_n_out_range2,
                                n_particles*sizeof(int) ) ) ;
        CUDA_SAFE_CALL(
                    cudaMalloc( (void**)&dev_in_range,
                                total_features*sizeof(char) ) ) ;


        // copy inputs
        CUDA_SAFE_CALL(
            cudaMemcpy( dev_maps, &concat[0], total_features*sizeof(GaussianType),
                        cudaMemcpyHostToDevice )
        ) ;
        CUDA_SAFE_CALL(
                    cudaMemcpy( dev_map_sizes, &map_sizes[0], n_particles*sizeof(int),
                        cudaMemcpyHostToDevice )
        ) ;


        // kernel launch
        DEBUG_MSG("launching computeInRangeKernel") ;
        DEBUG_VAL(nThreads) ;
        computeInRangeKernel<<<n_particles,nThreads>>>
            ( dev_maps, dev_map_sizes, n_particles, dev_poses, dev_in_range,
              dev_n_in_range, dev_n_out_range2 ) ;
        CUDA_SAFE_THREAD_SYNC();

        // allocate outputs
        in_range.resize(total_features);

        // copy outputs
        CUDA_SAFE_CALL(
                    cudaMemcpy( &in_range[0],dev_in_range,
                                total_features*sizeof(char),
                                cudaMemcpyDeviceToHost )
        ) ;
        CUDA_SAFE_CALL(
            cudaMemcpy( &n_in_range_vec[0],dev_n_in_range,n_particles*sizeof(int),
                        cudaMemcpyDeviceToHost )
        ) ;
        CUDA_SAFE_CALL(
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
        CUDA_SAFE_CALL( cudaFree( dev_maps ) ) ;
        CUDA_SAFE_CALL( cudaFree( dev_in_range ) ) ;
        CUDA_SAFE_CALL( cudaFree( dev_n_in_range ) ) ;


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
    CUDA_SAFE_CALL(
                    cudaMalloc( (void**)&dev_maps_inrange,
                                n_in_range*sizeof(GaussianType) ) ) ;
    CUDA_SAFE_CALL(
                    cudaMalloc( (void**)&dev_map_offsets,
                                (n_particles+1)*sizeof(int) ) ) ;
    CUDA_SAFE_CALL(
                cudaMalloc((void**)&dev_maps_updated,
                           n_update*sizeof(GaussianType)) ) ;
    CUDA_SAFE_CALL(
                cudaMalloc((void**)&dev_merged_flags,
                           n_update*sizeof(bool)) ) ;

    // copy inputs
    CUDA_SAFE_CALL(
        cudaMemcpy( dev_maps_inrange, &features_in[0],
                    n_in_range*sizeof(GaussianType),
                    cudaMemcpyHostToDevice )
    ) ;
    CUDA_SAFE_CALL( cudaMemcpy( dev_map_offsets, &map_offsets_in[0],
                                (n_particles+1)*sizeof(int),
                                cudaMemcpyHostToDevice ) ) ;
}

template <class GaussianType>
void
mergeAndCopyMaps(GaussianType* dev_maps_updated,
                 bool* dev_merged_flags,
                 vector<GaussianType> features_out1,
                 vector<GaussianType> features_out2,
                 vector<int> n_in_range_vec,
                 vector<int> n_out_range1_vec,
                 vector<int> n_out_range2_vec,
                 int n_particles, int n_measure, int n_update,
                 vector<vector<GaussianType> >& maps_output )
{
    vector<int> map_sizes(n_particles) ;
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
    int* dev_map_sizes = NULL ;

    int n_out_range1 = features_out1.size() ;
    int n_out_range2 = features_out2.size() ;



    // recombine updated in-range map with nearly in-range map do merging
    DEBUG_MSG("Recombining maps") ;
    combined_size = (n_update+n_out_range2)*sizeof(GaussianType) ;
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_maps_combined, combined_size ) ) ;
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_merged_flags_combined,
                                (n_update+n_out_range2)*sizeof(bool) ) ) ;


    for ( int n = 0 ; n < n_particles ; n++ )
    {
        // in-range map for particle n
        int n_in_range_n = n_in_range_vec[n]*(n_measure+1)+n_measure ;
        CUDA_SAFE_CALL( cudaMemcpy( dev_maps_combined+offset,
                                    dev_maps_updated+offset_updated,
                                    n_in_range_n*sizeof(GaussianType),
                                    cudaMemcpyDeviceToDevice) ) ;
        CUDA_SAFE_CALL( cudaMemcpy( dev_merged_flags_combined+offset,
                                    dev_merged_flags+offset_updated,
                                    n_in_range_n*sizeof(bool)
                                    ,cudaMemcpyDeviceToDevice ) ) ;
        offset += n_in_range_n ;
        offset_updated += n_in_range_n ;

        // nearly in range map for particle n
        vector<char> merged_flags_out(n_out_range2_vec[n],0) ;
        CUDA_SAFE_CALL( cudaMemcpy( dev_maps_combined+offset,
                                    &features_out2[offset_out],
                                    n_out_range2_vec[n]*sizeof(GaussianType),
                                    cudaMemcpyHostToDevice ) ) ;
        CUDA_SAFE_CALL( cudaMemcpy( dev_merged_flags_combined+offset,
                                    &merged_flags_out[0],
                                    n_out_range2_vec[n]*sizeof(bool),
                                    cudaMemcpyHostToDevice) ) ;
        offset += n_out_range2_vec[n] ;
        offset_out += n_out_range2_vec[n] ;


        map_sizes[n] = n_out_range2_vec[n] +
                n_in_range_n ;
    }

    DEBUG_VAL(combined_size) ;
    CUDA_SAFE_CALL( cudaMalloc((void**)&dev_maps_merged,
                           combined_size ) ) ;
    CUDA_SAFE_CALL( cudaMalloc((void**)&dev_n_merged,
                               n_particles*sizeof(int) ) ) ;
    CUDA_SAFE_CALL( cudaMalloc((void**)&dev_map_sizes,
                               n_particles*sizeof(int) ) ) ;
    CUDA_SAFE_CALL( cudaMemcpy( dev_map_sizes, &map_sizes[0],
                                n_particles*sizeof(int),
                                cudaMemcpyHostToDevice ) ) ;
    CUDA_SAFE_THREAD_SYNC() ;



    DEBUG_MSG("launching phdUpdateMergeKernel") ;
    phdUpdateMergeKernel<<<n_particles,256>>>
        ( dev_maps_combined, dev_maps_merged, dev_n_merged,
          dev_merged_flags_combined, dev_map_sizes, n_particles ) ;
    CUDA_SAFE_THREAD_SYNC() ;

    // copy one feature and look at it
    GaussianType feature_test ;
    CUDA_SAFE_CALL(cudaMemcpy(&feature_test,dev_maps_merged,sizeof(GaussianType),cudaMemcpyDeviceToHost) ) ;
    cout << "first merged feature: " << endl ;
    cout << feature_test.weight << " " << feature_test.mean[0]
         << " " << feature_test.mean[1] << " " << feature_test.cov[0]
         << " " << feature_test.cov[1] << " " << feature_test.cov[2]
         << " " << feature_test.cov[3] << endl ;

    // allocate outputs
    DEBUG_MSG("Allocating update and merge outputs") ;
    maps_merged = (GaussianType*)malloc( combined_size ) ;
    map_sizes_merged = (int*)malloc( n_particles*sizeof(int) ) ;

    // copy outputs
    CUDA_SAFE_CALL(
                cudaMemcpy( maps_merged, dev_maps_merged,
                            combined_size,
                            cudaMemcpyDeviceToHost ) ) ;
    CUDA_SAFE_CALL(
                cudaMemcpy( map_sizes_merged, dev_n_merged,
                            n_particles*sizeof(int),
                            cudaMemcpyDeviceToHost ) ) ;

    offset_updated = 0 ;
    offset_out = 0 ;
    for ( int i = 0 ; i < n_particles ; i++ )
    {
//        DEBUG_VAL(map_sizes_merged[i]) ;
        maps_output[i].assign(maps_merged+offset_updated,
                            maps_merged+offset_updated+map_sizes_merged[i]) ;
        offset_updated += map_sizes[i] ;

        // recombine with out-of-range features, if any
        if ( n_out_range1 > 0 && n_out_range1_vec[i] > 0 )
        {
            maps_output[i].insert( maps_output[i].end(),
                                    features_out1.begin()+offset_out,
                                    features_out1.begin()+offset_out+n_out_range1_vec[i] ) ;
            offset_out += n_out_range1_vec[i] ;
        }
    }

    free(maps_merged) ;
    free(map_sizes_merged) ;
    CUDA_SAFE_CALL( cudaFree( dev_maps_combined ) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_maps_merged ) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_merged_flags_combined ) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_n_merged ) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_maps_updated) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_merged_flags) ) ;
}

ParticleSLAM
phdUpdate(ParticleSLAM& particles, measurementSet measurements)
{
    //------- Variable Declarations ---------//

    int n_measure = 0 ;
    int n_particles = particles.n_particles ;
    vector<int> map_sizes_static(n_particles,0) ;
    vector<int> map_sizes_dynamic(n_particles,0) ;    

    // map offsets
    vector<int> map_offsets_in_static(n_particles+1,0) ;
    vector<int> map_offsets_out_static(n_particles+1,0) ;

    ParticleSLAM particlesPreMerge ;

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
    CUDA_SAFE_CALL(
                cudaMemcpyToSymbol( Z, &measurements[0],
                                    n_measure*sizeof(RangeBearingMeasurement) ) ) ;

    // copy particle poses to device
    CUDA_SAFE_CALL(
            cudaMalloc( (void**)&dev_poses,
                        n_particles*sizeof(ConstantVelocityState) ) ) ;
    CUDA_SAFE_CALL(
                cudaMemcpy(dev_poses,&particles.states[0],
                           n_particles*sizeof(ConstantVelocityState),
                           cudaMemcpyHostToDevice) ) ;

    // extract in-range portions of maps, and allocate output arrays
    prepareUpdateInputs( particlesPreMerge.maps_static,
                         dev_poses, n_particles, n_measure,
                         dev_maps_inrange_static, dev_map_offsets_static,
                         dev_maps_updated_static, dev_merged_flags_static,
                         features_in_static, features_out1_static,
                         features_out2_static, n_in_range_vec_static,
                         n_out_range1_vec_static, n_out_range2_vec_static) ;
    if(config.dynamicFeatures)
    {
        prepareUpdateInputs( particlesPreMerge.maps_dynamic,
                             dev_poses, n_particles, n_measure,
                             dev_maps_inrange_dynamic, dev_map_offsets_dynamic,
                             dev_maps_updated_dynamic, dev_merged_flags_dynamic,
                             features_in_dynamic, features_out1_dynamic,
                             features_out2_dynamic, n_in_range_vec_dynamic,
                             n_out_range1_vec_dynamic,n_out_range2_vec_dynamic) ;
    }

    // allocate arrays for particle weight update
    CUDA_SAFE_CALL(
                cudaMalloc((void**)&dev_particle_weights,
                           n_particles*sizeof(REAL) ) ) ;


    // launch kernel
    int nBlocks = min(n_particles,32768) ;
    int n_update_static = features_in_static.size()*(n_measure+1)
            + n_measure*n_particles ;
    int n_update_dynamic = features_in_dynamic.size()*(n_measure+1)
            + n_measure*n_particles ;

    cudaPrintfInit(4194304) ;
    if(config.dynamicFeatures)
    {
        DEBUG_MSG("launching phdUpdateKernelMixed") ;
        phdUpdateKernelMixed<<<nBlocks,256>>>(
            dev_poses, dev_maps_inrange_static, dev_maps_inrange_dynamic,
            dev_map_offsets_static, dev_map_offsets_dynamic,
            n_particles,n_measure,
            dev_maps_updated_static, dev_maps_updated_dynamic,
            dev_merged_flags_static, dev_merged_flags_dynamic,
            dev_particle_weights);
        CUDA_SAFE_THREAD_SYNC() ;
        CUDA_SAFE_CALL( cudaFree( dev_maps_inrange_dynamic ) ) ;
        CUDA_SAFE_CALL( cudaFree( dev_map_offsets_dynamic ) ) ;
    }
    else
    {
        DEBUG_MSG("launching phdUpdateKernelStatic") ;
        phdUpdateKernelStatic<<<nBlocks,256>>>(
            dev_poses, dev_maps_inrange_static,dev_map_offsets_static,
            n_particles,n_measure,dev_maps_updated_static,
            dev_merged_flags_static,dev_particle_weights) ;
        CUDA_SAFE_THREAD_SYNC() ;
    }
    cudaPrintfDisplay(stdout,false) ;
    cudaPrintfEnd();

    CUDA_SAFE_CALL( cudaFree( dev_maps_inrange_static ) ) ; 
    CUDA_SAFE_CALL( cudaFree( dev_map_offsets_static ) ) ;


    // check input weights against merge flags
    cout << "DEBUG first updated feature" << endl ;
    bool* merged_flags = (bool*)malloc(n_update_static*sizeof(bool)) ;
    Gaussian2D* maps_updated = (Gaussian2D*)malloc( n_update_static*sizeof(Gaussian2D) ) ;
    cudaMemcpy( merged_flags, dev_merged_flags_static, n_update_static*sizeof(bool),cudaMemcpyDeviceToHost) ;
    CUDA_SAFE_CALL(
                cudaMemcpy( maps_updated, dev_maps_updated_static,
                            n_update_static*sizeof(Gaussian2D),
                            cudaMemcpyDeviceToHost ) ) ;
//    for (int j = 0 ; j < n_update_static ; j++)
//    {
//        cout << "(" << maps_updated[j].weight << " | " << merged_flags[j] << ")" << endl ;
//    }
    print_feature(maps_updated[0]) ;
    print_feature(maps_updated[1]) ;
    free(maps_updated) ;
    free(merged_flags) ;


    /******************************************************
     *
     * Merge updated maps and copy back to host
     *
     ******************************************************/
    mergeAndCopyMaps( dev_maps_updated_static,dev_merged_flags_static,
                      features_out1_static,
                      features_out2_static, n_in_range_vec_static,
                      n_out_range1_vec_static,
                      n_out_range2_vec_static, n_particles,
                      n_measure,n_update_static, particles.maps_static ) ;

    if(config.dynamicFeatures)
    {
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
    REAL* particle_weights = (REAL*)malloc(n_particles*sizeof(REAL)) ;
    CUDA_SAFE_CALL( cudaMemcpy(particle_weights,dev_particle_weights,
                               n_particles*sizeof(REAL),
                               cudaMemcpyDeviceToHost ) ) ;
    // multiply weights be multi-object likelihood
    for ( int i = 0 ; i < n_particles ; i++ )
    {
        particles.weights[i] += particle_weights[i]  ;
    }

    // normalize
    REAL weightSum = logSumExp(particles.weights) ;
    DEBUG_VAL(weightSum) ;
    for (int i = 0 ; i < n_particles ; i++ )
    {
        particles.weights[i] -= weightSum ;
    }

    // free memory
    CUDA_SAFE_CALL( cudaFree( dev_particle_weights ) ) ;
    free(particle_weights) ;
    CUDA_SAFE_CALL( cudaFree( dev_poses ) ) ;
    return particlesPreMerge ;
}

ParticleSLAM resampleParticles( ParticleSLAM oldParticles, int n_new_particles)
{
    if ( n_new_particles < 0 )
    {
        n_new_particles = oldParticles.n_particles ;
    }
    ParticleSLAM newParticles(n_new_particles) ;
    vector<int> idx_resample(n_new_particles) ;
    double interval = 1.0/n_new_particles ;
    double r = randu01() * interval ;
    double c = exp(oldParticles.weights[0]) ;
    idx_resample.resize(n_new_particles, 0) ;
    int i = 0 ;
//	DEBUG_VAL(interval) ;
    for ( int j = 0 ; j < n_new_particles ; j++ )
    {
        r = j*interval + randu01()*interval ;
        while( r > c )
        {
            i++ ;
            // sometimes the weights don't exactly add up to 1, so i can run
            // over the indexing bounds. When this happens, find the most highly
            // weighted particle and fill the rest of the new samples with it
            if ( i >= oldParticles.n_particles || i < 0 || isnan(i) )
            {
                DEBUG_VAL(r) ;
                DEBUG_VAL(c) ;
                double max_weight = -1 ;
                int max_idx = -1 ;
                for ( int k = 0 ; k < oldParticles.n_particles ; k++ )
                {
                    DEBUG_MSG("Warning: particle weights don't add up to 1!s") ;
                    if ( exp(oldParticles.weights[k]) > max_weight )
                    {
                        max_weight = exp(oldParticles.weights[k]) ;
                        max_idx = k ;
                    }
                }
                i = max_idx ;
                // set c = 2 so that this while loop is never entered again
                c = 2 ;
                break ;
            }
            c += exp(oldParticles.weights[i]) ;
        }
//		DEBUG_VAL(i) ;
        newParticles.weights[j] = safeLog(interval) ;
        newParticles.states[j] = oldParticles.states[i] ;
        newParticles.maps_static[j] = oldParticles.maps_static[i] ;
        newParticles.maps_dynamic[j] = oldParticles.maps_dynamic[i] ;
        newParticles.cardinalities[j] = oldParticles.cardinalities[i] ;
        idx_resample[j] = i ;
//        r += interval ;
    }
    newParticles.resample_idx = idx_resample ;
    return newParticles ;
}

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
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_maps_in,
                                total_features*sizeof(GaussianType) ) ) ;
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_maps_out,
                                total_features*sizeof(GaussianType) ) ) ;
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_merged_sizes,
                                n_particles*sizeof(int) ) ) ;
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_map_sizes,
                                n_particles*sizeof(int) ) ) ;
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_merged_flags,
                                total_features*sizeof(bool) ) ) ;
    for ( int n = n_particles/2 ; n > 0 ; n >>= 1 )
    {
        DEBUG_VAL(n) ;
        for ( int i = 0 ; i < n ; i++ )
            map_sizes[i] = merged_sizes[2*i] + merged_sizes[2*i+1] ;
        CUDA_SAFE_CALL( cudaMemcpy( dev_map_sizes, map_sizes,
                                    n*sizeof(int),
                                    cudaMemcpyHostToDevice ) ) ;
        CUDA_SAFE_CALL( cudaMemcpy( dev_maps_in, all_features,
                                    total_features*sizeof(GaussianType),
                                    cudaMemcpyHostToDevice) ) ;
        CUDA_SAFE_CALL( cudaMemcpy( dev_merged_flags, merged_flags,
                                    total_features*sizeof(bool),
                                    cudaMemcpyHostToDevice)) ;
        // kernel launch
        phdUpdateMergeKernel<<<n,256>>>( dev_maps_in, dev_maps_out, dev_merged_sizes,
                                         dev_merged_flags, dev_map_sizes, n ) ;

        CUDA_SAFE_CALL( cudaMemcpy( maps_out, dev_maps_out,
                                    total_features*sizeof(GaussianType),
                                    cudaMemcpyDeviceToHost) ) ;
        CUDA_SAFE_CALL( cudaMemcpy( merged_sizes, dev_merged_sizes,
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

    CUDA_SAFE_CALL( cudaFree( dev_maps_in ) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_maps_out ) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_merged_sizes ) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_merged_flags ) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_map_sizes ) ) ;
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

void
recoverSlamState(ParticleSLAM particles, ConstantVelocityState& expectedPose,
        vector<Gaussian2D>& expected_map_static, vector<Gaussian4D>& expected_map_dynamic,
        vector<REAL>& cn_estimate )
{
    if ( particles.n_particles > 1 )
    {
        // calculate the weighted mean of the particle poses
        expectedPose.px = 0 ;
        expectedPose.py = 0 ;
        expectedPose.ptheta = 0 ;
        expectedPose.vx = 0 ;
        expectedPose.vy = 0 ;
        expectedPose.vtheta = 0 ;
        for ( int i = 0 ; i < particles.n_particles ; i++ )
        {
            double exp_weight = exp(particles.weights[i]) ;
            expectedPose.px += exp_weight*particles.states[i].px ;
            expectedPose.py += exp_weight*particles.states[i].py ;
            expectedPose.ptheta += exp_weight*particles.states[i].ptheta ;
            expectedPose.vx += exp_weight*particles.states[i].vx ;
            expectedPose.vy += exp_weight*particles.states[i].vy ;
            expectedPose.vtheta += exp_weight*particles.states[i].vtheta ;
        }
//		gaussianMixture tmpMap = computeExpectedMap(particles) ;
//		tmpMap.erase(
//						remove_if( tmpMap.begin(), tmpMap.end(),
//								expectedFeaturesPredicate),
//						tmpMap.end() ) ;
//		*expectedMap = tmpMap ;

        // Maximum a priori estimate
        double max_weight = -FLT_MAX ;
        int max_idx = -1 ;
        for ( int i = 0 ; i < particles.n_particles ; i++ )
        {
            if ( particles.weights[i] > max_weight )
            {
                max_idx = i ;
                max_weight = particles.weights[i] ;
            }
        }
        DEBUG_VAL(max_idx) ;
//		expectedPose = particles.states[max_idx] ;

        if ( config.mapEstimate == 0)
        {
            expected_map_static = particles.maps_static[max_idx] ;
            expected_map_dynamic = particles.maps_dynamic[max_idx] ;
        }
        else
        {
//            expectedMap = computeExpectedMap( particles ) ;
        }

        cn_estimate = particles.cardinalities[max_idx] ;
    }
    else
    {
//        vector<GaussianType> tmpMap( particles.maps[0] ) ;
//        tmpMap.erase(
//                remove_if( tmpMap.begin(), tmpMap.end(),
//                        expectedFeaturesPredicate),
//                tmpMap.end() ) ;
        expectedPose = particles.states[0] ;
        expected_map_static = particles.maps_static[0] ;
        expected_map_dynamic = particles.maps_dynamic[0] ;
        cn_estimate = particles.cardinalities[0] ;
    }
}

/// copy the configuration structure to constant device memory
void
setDeviceConfig( const SlamConfig& config )
{
    CUDA_SAFE_CALL(cudaMemcpyToSymbol( dev_config, &config, sizeof(SlamConfig) ) ) ;
//    seed_rng();
}

//--- explicitly instantiate templates so linking works correctly

//--- End template specialization
