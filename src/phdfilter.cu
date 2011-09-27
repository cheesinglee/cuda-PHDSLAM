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
#include "slamtypes.h"
//#include "slamparams.h"
#include <cutil.h>
#include <complex.h>
#include <fftw3.h>
#include <assert.h>
#include <float.h>
//#include "cuPrintf.cu"

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
#define DEBUG_MSG(x) cout <<"["<< __func__ << "]: " << x << endl
#define DEBUG_VAL(x) cout << "["<<__func__ << "]: " << #x << " = " << x << endl ;
#else
#define DEBUG_MSG(x)
#define DEBUG_VAL(x)
#endif

//--- Make kernel helper functions externally visible
extern "C"
void
initCphdConstants() ;

extern "C"
void
phdPredict(ParticleSLAM& particles, AckermanControl control ) ;

extern "C"
void
phdPredictVp( ParticleSLAM& particles ) ;

extern "C"
void
addBirths( ParticleSLAM& particles, measurementSet ZPrev ) ;

extern "C"
ParticleSLAM
phdUpdate(ParticleSLAM& particles, measurementSet measurements) ;

extern "C"
ParticleSLAM resampleParticles( ParticleSLAM oldParticles, int nParticles=-1 ) ;

extern "C"
void recoverSlamState(ParticleSLAM particles, ConstantVelocityState& expectedPose,
        gaussianMixture& expectedMap, vector<REAL>& cn_estimate ) ;

extern "C"
void setDeviceConfig( const SlamConfig& config ) ;
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
//__device__ REAL* dev_qspower ;
//__device__ REAL* dev_pspower ;
REAL* dev_cn_clutter ;

//ConstantVelocityModelProps modelProps  = {STDX, STDY,STDTHETA} ;
//ConstantVelocity2DKinematicModel motionModel(modelProps) ;
__host__ std::vector<REAL>
conv(std::vector<REAL> a, std::vector<REAL> b)
{
    int m = a.size() ;
    int n = b.size() ;
    int len = m + n - 1 ;
    std::vector<REAL> c(len) ;
    std::fill( c.begin(),c.end(),0) ;
    for ( int k = 0 ; k < len ; k++ )
    {
        int start_idx = max(0,k-n+1) ;
        int stop_idx = min(k,m-1) ;
        for (int j = start_idx ; j <= stop_idx ; j++ )
        {
            c[k] += a[j]*b[k-j] ;
        }
    }
    return c ;
}

__host__ __device__ REAL
wrapAngle(REAL a)
{
    REAL remainder = fmod(a, REAL(2*M_PI)) ;
    if ( remainder > M_PI )
        remainder -= 2*M_PI ;
    else if ( remainder < -M_PI )
        remainder += 2*M_PI ;
    return remainder ;
}

__device__ void
sumByReduction( volatile REAL* sdata, REAL mySum, const unsigned int tid )
{
    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads();
    if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads();

    if (tid < 32)
    {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
        sdata[tid] = mySum = mySum + sdata[tid + 16];
        sdata[tid] = mySum = mySum + sdata[tid +  8];
        sdata[tid] = mySum = mySum + sdata[tid +  4];
        sdata[tid] = mySum = mySum + sdata[tid +  2];
        sdata[tid] = mySum = mySum + sdata[tid +  1];
    }
    __syncthreads() ;
}

__device__ void
productByReduction( volatile REAL* sdata, REAL mySum, const unsigned int tid )
{
    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    if (tid < 128) { sdata[tid] = mySum = mySum * sdata[tid + 128]; } __syncthreads();
    if (tid <  64) { sdata[tid] = mySum = mySum * sdata[tid +  64]; } __syncthreads();

    if (tid < 32)
    {
        sdata[tid] = mySum = mySum * sdata[tid + 32];
        sdata[tid] = mySum = mySum * sdata[tid + 16];
        sdata[tid] = mySum = mySum * sdata[tid +  8];
        sdata[tid] = mySum = mySum * sdata[tid +  4];
        sdata[tid] = mySum = mySum * sdata[tid +  2];
        sdata[tid] = mySum = mySum * sdata[tid +  1];
    }
    __syncthreads() ;
}

__device__ void
maxByReduction( volatile REAL* sdata, REAL val, const unsigned int tid )
{
    sdata[tid] = val ;
    __syncthreads();

    // do reduction in shared mem
    if (tid < 128) { sdata[tid] = val = fmax(sdata[tid+128],val) ; } __syncthreads();
    if (tid <  64) { sdata[tid] = val = fmax(sdata[tid+64],val) ; } __syncthreads();

    if (tid < 32)
    {
        sdata[tid] = val = fmax(sdata[tid+32],val) ;
        sdata[tid] = val = fmax(sdata[tid+16],val) ;
        sdata[tid] = val = fmax(sdata[tid+8],val) ;
        sdata[tid] = val = fmax(sdata[tid+4],val) ;
        sdata[tid] = val = fmax(sdata[tid+2],val) ;
        sdata[tid] = val = fmax(sdata[tid+1],val) ;
    }
    __syncthreads() ;
}

__device__ void
makePositiveDefinite( REAL A[4] )
{
    // eigenvalues:
    REAL detA = A[0]*A[3] + A[1]*A[2] ;
    // check if already positive definite
    if ( detA > 0 && A[0] > 0 )
    {
        A[1] = (A[1] + A[2])/2 ;
        A[2] = A[1] ;
        return ;
    }
    REAL trA = A[0] + A[3] ;
    REAL trA2 = trA*trA ;
    REAL eval1 = 0.5*trA + 0.5*sqrt( trA2 - 4*detA ) ;
    REAL eval2 = 0.5*trA - 0.5*sqrt( trA2 - 4*detA ) ;

    // eigenvectors:
    REAL Q[4] ;
    if ( fabs(A[1]) > 0 )
    {
        Q[0] = eval1 - A[3] ;
        Q[1] = A[1] ;
        Q[2] = eval2 - A[3] ;
        Q[3] = A[1] ;
    }
    else if ( fabs(A[2]) > 0 )
    {
        Q[0] = A[2] ;
        Q[1] = eval1 - A[0] ;
        Q[2] = A[2] ;
        Q[3] = eval2 - A[0] ;
    }
    else
    {
        Q[0] = 1 ;
        Q[1] = 0 ;
        Q[2] = 0 ;
        Q[3] = 1 ;
    }

    // make eigenvalues positive
    if ( eval1 < 0 )
        eval1 = DBL_EPSILON ;
    if ( eval2 < 0 )
        eval2 = DBL_EPSILON ;

    // compute the approximate matrix
    A[0] = Q[0]*Q[0]*eval1 + Q[2]*Q[2]*eval2 ;
    A[1] = Q[0]*eval1*Q[1] + Q[2]*eval2*Q[3] ;
    A[2] = A[1] ;
    A[3] = Q[1]*Q[1]*eval1 + Q[3]*Q[3]*eval2 ;
}

__device__ REAL
computeMahalDist(Gaussian2D a, Gaussian2D b)
{
    REAL innov[2] ;
    REAL sigma[4] ;
    REAL detSigma ;
    REAL sigmaInv[4] = {1,0,0,1} ;
    innov[0] = a.mean[0] - b.mean[0] ;
    innov[1] = a.mean[1] - b.mean[1] ;
//    sigma[0] = a.cov[0] + b.cov[0] ;
//    sigma[1] = a.cov[1] + b.cov[1] ;
//    sigma[2] = a.cov[2] + b.cov[2] ;
//    sigma[3] = a.cov[3] + b.cov[3] ;
//    sigma[0] = a.cov[0] ;
//    sigma[1] = a.cov[1] ;
//    sigma[2] = a.cov[2] ;
//    sigma[3] = a.cov[3] ;
    sigma[0] = b.cov[0] ;
    sigma[1] = b.cov[1] ;
    sigma[2] = b.cov[2] ;
    sigma[3] = b.cov[3] ;
    detSigma = sigma[0]*sigma[3] - sigma[1]*sigma[2] ;
//	detSigma = a.cov[0]*a.cov[3] - a.cov[1]*a.cov[2] ;
    if (detSigma > FLT_MIN)
    {
//		sigmaInv[0] = a.cov[3]/detSigma ;
//		sigmaInv[1] = -a.cov[1]/detSigma ;
//		sigmaInv[2] = -a.cov[2]/detSigma ;
//		sigmaInv[3] = a.cov[0]/detSigma ;
        sigmaInv[0] = sigma[3]/detSigma ;
        sigmaInv[1] = -sigma[1]/detSigma ;
        sigmaInv[2] = -sigma[2]/detSigma ;
        sigmaInv[3] = sigma[0]/detSigma ;
    }
    return  innov[0]*innov[0]*sigmaInv[0] +
            innov[0]*innov[1]*(sigmaInv[1]+sigmaInv[2]) +
            innov[1]*innov[1]*sigmaInv[3] ;
}

__device__ REAL
computeHellingerDist( Gaussian2D a, Gaussian2D b)
{
    REAL innov[2] ;
    REAL sigma[4] ;
    REAL detSigma ;
    REAL sigmaInv[4] = {1,0,0,1} ;
    REAL dist ;
    innov[0] = a.mean[0] - b.mean[0] ;
    innov[1] = a.mean[1] - b.mean[1] ;
    sigma[0] = a.cov[0] + b.cov[0] ;
    sigma[1] = a.cov[1] + b.cov[1] ;
    sigma[2] = a.cov[2] + b.cov[2] ;
    sigma[3] = a.cov[3] + b.cov[3] ;
    detSigma = sigma[0]*sigma[3] - sigma[1]*sigma[2] ;
    if (detSigma > FLT_MIN)
    {
        sigmaInv[0] = sigma[3]/detSigma ;
        sigmaInv[1] = -sigma[1]/detSigma ;
        sigmaInv[2] = -sigma[2]/detSigma ;
        sigmaInv[3] = sigma[0]/detSigma ;
    }
    REAL epsilon = -0.25*
            (innov[0]*innov[0]*sigmaInv[0] +
             innov[0]*innov[1]*(sigmaInv[1]+sigmaInv[2]) +
             innov[1]*innov[1]*sigmaInv[3]) ;

    // determinant of half the sum of covariances
    detSigma /= 4 ;
    dist = 1/detSigma ;

    // product of covariances
    sigma[0] = a.cov[0]*b.cov[0] + a.cov[2]*b.cov[1] ;
    sigma[1] = a.cov[1]*b.cov[0] + a.cov[3]*b.cov[1] ;
    sigma[2] = a.cov[0]*b.cov[2] + a.cov[2]*b.cov[3] ;
    sigma[3] = a.cov[1]*b.cov[2] + a.cov[3]*b.cov[3] ;
    detSigma = sigma[0]*sigma[3] - sigma[1]*sigma[2] ;
    dist *= sqrt(detSigma) ;
    dist = 1 - sqrt(dist)*exp(epsilon) ;
    return dist ;
}

/// a nan-safe logarithm
__device__ __host__
REAL safeLog( REAL x )
{
    if ( x <= 0 )
        return LOG0 ;
    else
        return log(x) ;
}


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

void
initCphdConstants()
{
    vector<REAL> log_factorials( config.maxCardinality+1) ;
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
    const int prior_idx = floor((float)predict_idx/dev_config.nPredictParticles) ;
    ConstantVelocityState oldState = particles_prior[prior_idx] ;
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
    particles_predict[predict_idx] = newState ;
}

__global__ void
phdPredictKernel(ConstantVelocityState* particles_prior,
        ConstantVelocityNoise* noise, ConstantVelocityState* particles_predict )
{
    const int tid = threadIdx.x ;
    const int predict_idx = blockIdx.x*blockDim.x + tid ;
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
//		REAL innersum = 0 ;
//		for ( int l = j ; l <= dev_config.maxCardinality ; l++ )
//		{
//			int c_idx = (dev_config.maxCardinality+1)*l + j ;
//			innersum += exp(dev_C[c_idx]+cn_prior_shared[l]) ;
//		}
        outersum += exp(cn_births[n-j]+cn_prior_shared[j]) ;
    }
    if ( outersum != 0)
        cn_predict[cn_offset+n] = safeLog(outersum) ;
    else
        cn_predict[cn_offset+n] = LOG0 ;
}

void
phdPredict(ParticleSLAM& particles, AckermanControl control )
{
    // start timer
    cudaEvent_t start, stop ;
    cudaEventCreate( &start ) ;
    cudaEventCreate( &stop ) ;
    cudaEventRecord( start,0 ) ;

    int nParticles = particles.nParticles ;
    int nPredict = nParticles*config.nPredictParticles ;

    /////////////////////////////////////////////////////////////
    //
    // do the cardinality prediction for the CPHD filter
    //
    /////////////////////////////////////////////////////////////
    if ( config.filterType == CPHD_TYPE )
    {
        // concatenate the cardinalities of all the particles
        vector<REAL> cn_concat ;
        for ( int i = 0 ; i < nParticles ; i++ )
        {
            cn_concat.insert( cn_concat.end(), particles.cardinalities[i].begin(),
                              particles.cardinalities[i].end() ) ;
        }

        // allocate memory
        REAL* dev_cn_prior = NULL ;
        REAL* dev_cn_births = NULL ;
        REAL* dev_cn_predict = NULL ;
        size_t cn_size = (config.maxCardinality + 1)*sizeof(REAL) ;
        CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_cn_prior, nParticles*cn_size ) ) ;
        CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_cn_births, cn_size ) ) ;
        CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_cn_predict, nParticles*cn_size ) ) ;

        // copy inputs
        CUDA_SAFE_CALL( cudaMemcpy( dev_cn_prior, &cn_concat[0],
                                    nParticles*cn_size, cudaMemcpyHostToDevice ) ) ;
        CUDA_SAFE_CALL( cudaMemcpy( dev_cn_births,
                                    &particles.cardinality_birth[0],
                                    cn_size, cudaMemcpyHostToDevice) ) ;

        // launch the kernel
        size_t shmem_size = (config.maxCardinality+1)*sizeof(REAL) ;
        cardinalityPredictKernel<<<nParticles,config.maxCardinality+1, shmem_size>>>
                            ( dev_cn_prior, dev_cn_births, dev_C,
                              dev_cn_predict ) ;

        // copy outputs
        REAL* cn_predict = (REAL*)malloc( nParticles*cn_size ) ;
        CUDA_SAFE_CALL( cudaMemcpy( cn_predict, dev_cn_predict,
                                    nParticles*cn_size,
                                    cudaMemcpyDeviceToHost) ) ;
        int offset = 0 ;
        for ( int i = 0 ; i < nParticles ; i++ )
        {
            particles.cardinalities[i].assign( cn_predict+offset,
                                               cn_predict+offset+config.maxCardinality+1 ) ;
            offset += config.maxCardinality + 1 ;
        }
        free(cn_predict) ;
        cudaFree( dev_cn_prior ) ;
        cudaFree( dev_cn_births ) ;
        cudaFree( dev_cn_predict ) ;
    }

    // allocate device memory
    ConstantVelocityState* dev_states_prior = NULL ;
    ConstantVelocityState* dev_states_predict = NULL ;
    CUDA_SAFE_CALL(
                cudaMalloc((void**)&dev_states_prior,
                           nParticles*sizeof(ConstantVelocityState) ) ) ;
    CUDA_SAFE_CALL(
                cudaMalloc((void**)&dev_states_predict,
                           nPredict*sizeof(ConstantVelocityState) ) ) ;

    // copy inputs
    CUDA_SAFE_CALL(
                cudaMemcpy(dev_states_prior, &particles.states[0],
                           nParticles*sizeof(ConstantVelocityState),
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
                               nParticles*sizeof(ConstantVelocityNoise) ) ) ;
        CUDA_SAFE_CALL(
                    cudaMemcpy(dev_noise, &noiseVector[0],
                               nParticles*sizeof(ConstantVelocityNoise),
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
        vector<gaussianMixture> maps_predict ;
        vector<double> weights_predict ;
        vector< vector <REAL> > cardinalities_predict ;
        maps_predict.clear();
        maps_predict.reserve(nPredict);
        weights_predict.clear();
        weights_predict.reserve(nPredict);
        cardinalities_predict.clear();
        cardinalities_predict.reserve(nPredict);
        for ( int i = 0 ; i < nParticles ; i++ )
        {
            maps_predict.insert( maps_predict.end(), config.nPredictParticles,
                                 particles.maps[i] ) ;
            cardinalities_predict.insert( cardinalities_predict.end(),
                                          config.nPredictParticles,
                                          particles.cardinalities[i] ) ;
            weights_predict.insert( weights_predict.end(), config.nPredictParticles,
                                    particles.weights[i] - safeLog(config.nPredictParticles) ) ;
        }
//        DEBUG_VAL(maps_predict.size()) ;
        particles.maps = maps_predict ;
        particles.weights = weights_predict ;
        particles.cardinalities = cardinalities_predict ;
        particles.nParticles = nPredict ;
    }

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

__global__ void
birthsKernel( ConstantVelocityState* particles, int nParticles,
        RangeBearingMeasurement* ZZ, char* compatibleZ, REAL* dev_C,
        Gaussian2D* births, REAL* cn_birth )
{
//    __shared__ unsigned int birthCounter ;
    int tid = threadIdx.x ;
//    if ( tid == 0 )
//        birthCounter = 0 ;
    __syncthreads() ;
    int stateIdx = blockIdx.x  ;
    int offset = stateIdx*blockDim.x ;
    int birthIdx = offset + tid ;
    ConstantVelocityState s = particles[stateIdx] ;
//    if ( !compatibleZ[offset+tid] )
//    {
//        unsigned int birthIdx = atomicAdd(&birthCounter, 1) ;
//        birthIdx += offset ;
        RangeBearingMeasurement z = ZZ[tid] ;
        REAL theta = z.bearing + s.ptheta ;
        REAL dx = z.range*cos(theta) ;
        REAL dy = z.range*sin(theta) ;
        REAL J[4] ;
        J[0] = 	dx/z.range ;
        J[1] = dy/z.range ;
        J[2] = -dy ;
        J[3] = dx ;
        births[birthIdx].mean[0] = s.px + dx ;
        births[birthIdx].mean[1] = s.py + dy ;
        births[birthIdx].cov[0] = J[0]*J[0]*pow(dev_config.stdRange*dev_config.birthNoiseFactor,2)
                + J[2]*J[2]*pow(dev_config.stdBearing*dev_config.birthNoiseFactor,2) ;
        births[birthIdx].cov[1] = J[0]*pow(dev_config.stdRange*dev_config.birthNoiseFactor,2)*J[1]
                + J[2]*pow(dev_config.stdBearing*dev_config.birthNoiseFactor,2)*J[3] ;
        births[birthIdx].cov[2] = births[birthIdx].cov[1] ;
        births[birthIdx].cov[3] = J[1]*J[1]*pow(dev_config.stdRange*dev_config.birthNoiseFactor,2)
                + J[3]*J[3]*pow(dev_config.stdBearing*dev_config.birthNoiseFactor,2) ;
    //	makePositiveDefinite( births[birthIdx].cov) ;
        births[birthIdx].weight = dev_config.birthWeight ;
//    }

    // thread block 0 computes the birth cardinality for the CPHD filter
    // the birth cardinality is a binomial distribution B(k;n,p), where
    // n is the total number of birth measurements, p is the birth weight
    if ( dev_config.filterType == CPHD_TYPE && blockIdx.x == 0 )
    {
        int n = blockDim.x ;
        for ( int k = tid ; k <= dev_config.maxCardinality ; k+=blockDim.x )
        {
            if ( k <= dev_config.maxCardinality )
            {
                int c_idx = (k)*(dev_config.maxCardinality+1) + n ;
                cn_birth[k] = dev_C[c_idx] + k*safeLog(dev_config.birthWeight)
                        + (n-k)*safeLog(1-dev_config.birthWeight) ;
            }
        }
    }
}

void addBirths(ParticleSLAM& particles, measurementSet measurements )
{
    cudaEvent_t start, stop ;
    cudaEventCreate( &start ) ;
    cudaEventCreate( &stop ) ;
    cudaEventRecord( start, 0 ) ;

    // allocate inputs on device
//	DEBUG_MSG("Allocating inputs") ;
    ConstantVelocityState* d_particles = NULL ;
    RangeBearingMeasurement* d_measurements = NULL ;
    char* devCompatibleZ = NULL ;
    REAL* dev_cn_birth = NULL ;
    vector<char> compatibleZ ;
    int n_measure = measurements.size() ;
    int nParticles = particles.nParticles ;
    if ( particles.compatibleZ.size() > 0 && config.gateBirths > 0 )
    {
        compatibleZ = particles.compatibleZ ;
    }
    else
    {
        compatibleZ.assign(nParticles*n_measure,0) ;
    }
    size_t particlesSize = nParticles*sizeof(ConstantVelocityState) ;
    size_t measurementsSize = n_measure*sizeof(RangeBearingMeasurement) ;
    CUDA_SAFE_CALL(cudaMalloc( (void**)&d_particles, particlesSize ) );
    CUDA_SAFE_CALL(cudaMalloc( (void**)&d_measurements, measurementsSize ) ) ;
    CUDA_SAFE_CALL(cudaMalloc( (void**)&devCompatibleZ, nParticles*n_measure*sizeof(char) ) ) ;

    CUDA_SAFE_CALL(
                cudaMemcpy( d_particles, &particles.states[0], particlesSize,
                            cudaMemcpyHostToDevice ) ) ;
    CUDA_SAFE_CALL(
                cudaMemcpy( d_measurements, &measurements[0], measurementsSize,
                            cudaMemcpyHostToDevice ) ) ;
    CUDA_SAFE_CALL(
                cudaMemcpy( devCompatibleZ, &compatibleZ[0],
                            nParticles*n_measure*sizeof(char),
                            cudaMemcpyHostToDevice ) ) ;
    if ( config.filterType == CPHD_TYPE )
    {
        size_t cn_size = (config.maxCardinality+1)*sizeof(REAL) ;
        CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_cn_birth, cn_size ) ) ;
    }

    // allocate outputs on device
//	DEBUG_MSG("Allocating outputs") ;
    Gaussian2D* d_births ;
    int nBirths = n_measure*nParticles ;
    size_t birthsSize = nBirths*sizeof(Gaussian2D) ;
    cudaMalloc( (void**)&d_births, birthsSize ) ;

    // call the kernel
//	DEBUG_MSG("Launching Kernel") ;
    birthsKernel<<<nParticles, n_measure>>>( d_particles, nParticles,
            d_measurements, devCompatibleZ, dev_C, d_births, dev_cn_birth ) ;

    // retrieve outputs from device
//	DEBUG_MSG("Saving birth terms") ;
    Gaussian2D *births = new Gaussian2D[nBirths] ;
    cudaMemcpy( births, d_births, birthsSize, cudaMemcpyDeviceToHost ) ;
    vector<Gaussian2D> birthVector(births, births + nBirths) ;
    vector<Gaussian2D>::iterator i ;
    int n = 0 ;
    for ( i = birthVector.begin() ; i != birthVector.end() ; i+= n_measure )
    {
//        int offset = n*n_measure ;
//        int nRealBirths = 0 ;
//        for ( int j = 0 ; j<n_measure ; j++ )
//        {
//            if ( !compatibleZ[offset+j] )
//                nRealBirths++ ;
//        }
//        particles.maps[n].insert(particles.maps[n].end(),i,i+nRealBirths) ;
        particles.maps[n].insert(particles.maps[n].end(),i,i+n_measure) ;
        n++ ;
    }

    if ( config.filterType == CPHD_TYPE )
    {
        CUDA_SAFE_CALL( cudaMemcpy(&particles.cardinality_birth[0], dev_cn_birth,
                        (config.maxCardinality+1)*sizeof(REAL),
                        cudaMemcpyDeviceToHost ) ) ;
    }
//	cout << "Map 1: " << endl ;
//	for ( int n = 0 ; n < particles->maps[1].size() ; n++ )
//	{
//		cout << "Weight: " << particles->maps[1][n].weight <<
//				" Mean: (" << particles->maps[1][n].mean[0] << "," << particles->maps[1][n].mean[1] << ")" <<
//				" Covariance: (" << particles->maps[1][n].cov[0] << "," << particles->maps[1][n].cov[1] << "," <<
//				particles->maps[1][n].cov[2] << "," << particles->maps[1][n].cov[3] << ")" << endl ;
//	}

    cudaEventRecord( stop, 0 ) ;
    cudaEventSynchronize( stop ) ;
    float elapsed ;
    cudaEventElapsedTime( &elapsed, start, stop ) ;
    fstream birthsTimeFile("birthtime.log", fstream::out|fstream::app ) ;
    birthsTimeFile << elapsed << endl ;
    birthsTimeFile.close() ;
    delete[] births ;
    cudaFree(d_particles) ;
    cudaFree(d_measurements) ;
    cudaFree(d_births) ;
    cudaFree(devCompatibleZ) ;
    cudaFree(dev_cn_birth) ;
}

/// determine which features are in range
/*!
  * Each thread block handles a single particle. The threads in the block
  * evaluate the range and bearing [blockDim] features in parallel, looping
  * through all of the particle's features.

    \param predictedFeatures Features from all particles concatenated into a
        single array
    \param mapSizes Number of features in each particle, so that the function
        knows where the boundaries are in predictedFeatures
    \param nParticles Total number of particles
    \param poses Array of particle poses
    \param inRange Pointer to boolean array that is filled by the function.
        For each feature in predictedFeatures that is in range of its
        respective particle, the corresponding entry in this array is set to
        true
    \param nInRange Pointer to integer array that is filled by the function.
        Should be allocated to have [nParticles] elements. Each entry
        represents the number of in range features for each particle.
  */
__global__ void
computeInRangeKernel( Gaussian2D *predictedFeatures, int* mapSizes, int nParticles,
                ConstantVelocityState* poses, char* inRange, int* nInRange )
{
    int tid = threadIdx.x ;

    // total number of predicted features per block
    int n_featuresBlock ;
    // number of inrange features in the particle
    __shared__ int nInRangeBlock ;
    // vehicle pose of the thread block
    ConstantVelocityState blockPose ;

    Gaussian2D feature ;
    for ( int p = 0 ; p < nParticles ; p += gridDim.x )
    {
        if ( p + blockIdx.x < nParticles )
        {
            int predict_offset = 0 ;
            // compute the indexing offset for this particle
            int map_idx = p + blockIdx.x ;
            for ( int i = 0 ; i < map_idx ; i++ )
                predict_offset += mapSizes[i] ;
            // particle-wide values
            if ( tid == 0 )
                nInRangeBlock = 0 ;
            blockPose = poses[map_idx] ;
            n_featuresBlock = mapSizes[map_idx] ;
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
                    if ( r <= dev_config.maxRange &&
                         fabs(bearing) <= dev_config.maxBearing )
                    {
                        atomicAdd( &nInRangeBlock, 1 ) ;
                        inRange[featureIdx] = 1 ;
                    }
                }
            }
            // store nInrange
            __syncthreads() ;
            if ( tid == 0 )
            {
                nInRange[map_idx] = nInRangeBlock ;
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

/// perform the gaussian mixture PHD update, and the reduce the resulting mixture
/**
  PHD update algorithm as in Vo & Ma 2006. Gaussian mixture reduction (merging),
  as in the clustering algorithm by Salmond 1990.
    \param inRangeFeatures Array of in-range Gaussians, with which the PHD
        update will be performed
    \param mapSizes Integer array of sizes of each particle's map
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
phdUpdateKernel(Gaussian2D *inRangeFeatures, int* map_offsets,
        int nParticles, int n_measure, ConstantVelocityState* poses,
        char* compatibleZ,Gaussian2D *updated_features,
        bool *mergedFlags, REAL *particleWeights )
{
    // shared memory variables
    __shared__ ConstantVelocityState pose ;
    __shared__ REAL sdata[256] ;
//	__shared__ Gaussian2D maxFeature ;
//	__shared__ REAL wMerge ;
//	__shared__ REAL meanMerge[2] ;
//	__shared__ REAL covMerge[4] ;
//	__shared__ int mergedSize ;
    __shared__ REAL particle_weight ;
    __shared__ REAL cardinality_predict ;
    __shared__ REAL cardinality_updated ;
    __shared__ int map_idx ;
    __shared__ int update_offset ;
    __shared__ int n_features ;
    __shared__ int n_update ;
    __shared__ int predict_offset ;
    __shared__ int max_feature ;


    // initialize variables
    int tid = threadIdx.x ;
    // pre-update variables
    REAL featurePd = 0 ;
//    REAL dx, dy, r2, r, bearing ;
//    REAL J[4] = {0,0,0,0} ;
    REAL K[4] = {0,0,0,0} ;
//    REAL sigma[4] = {0,0,0,0} ;
    REAL sigmaInv[4] = {0,0,0,0} ;
//    REAL covUpdate[4] = {0,0,0,0} ;
    REAL detSigma = 0 ;
    Gaussian2D feature ;
    RangeBearingMeasurement z_predict ;
    REAL covUpdate[4] ;

    // update variables
    RangeBearingMeasurement z ;
    REAL innov[2] = {0,0} ;
    REAL meanUpdate[2] = {0,0} ;
    REAL dist = 0 ;
    REAL logLikelihood = 0 ;
    REAL w_partial = 0 ;
    REAL weightUpdate = 0 ;
    int updateIdx = 0 ;
    // loop over particles
    for ( int p = 0 ; p < nParticles ; p += gridDim.x )
    {
        if ( p + blockIdx.x < nParticles )
        {
            // initialize shared variables
            if ( tid == 0 )
            {
                map_idx = p + blockIdx.x ;
                predict_offset = map_offsets[map_idx] ;
                update_offset = predict_offset*(n_measure+1) ;
                n_features = map_offsets[map_idx+1] - map_offsets[map_idx] ;
                n_update = n_features*(n_measure+1) ;
                pose = poses[map_idx] ;
                particle_weight = 0 ;
    //			mergedSize = 0 ;
    //			mergedSizes[map_idx] = n_update ;
                cardinality_predict = 0.0 ;
                cardinality_updated = 0.0 ;
                max_feature = 0 ;
            }
            __syncthreads() ;

            if ( tid < n_measure )
                compatibleZ[map_idx*n_measure + tid] = 0 ;
            for ( int j = 0 ; j < n_features ; j += blockDim.x )
            {
                int feature_idx = j + tid ;
                if ( feature_idx < n_features )
                {
                    // get the feature corresponding to the current thread
                    feature = inRangeFeatures[predict_offset+feature_idx] ;

                    /*
                     * PRECOMPUTE UPDATE COMPONENTS
                     */


                    computePreUpdateComponents( pose, feature, K, covUpdate,
                                                &detSigma, sigmaInv, &featurePd,
                                                &z_predict ) ;

//                    // predicted measurement
//                    dx = feature.mean[0] - pose.px ;
//                    dy = feature.mean[1] - pose.py ;
//                    r2 = dx*dx + dy*dy ;
//                    r = sqrt(r2) ;
//                    bearing = wrapAngle(atan2f(dy,dx) - pose.ptheta) ;

//                    // probability of detection
//                    if ( r <= dev_config.maxRange && fabsf(bearing) <= dev_config.maxBearing )
//                        featurePd = dev_config.pd ;
//                    else
//                        featurePd = 0 ;

//                    // measurement jacobian wrt feature
//                    J[0] = dx/r ;
//                    J[2] = dy/r ;
//                    J[1] = -dy/r2 ;
//                    J[3] = dx/r2 ;

//                    // BEGIN Maple-Generated expressions
//            #define P feature.cov
//            #define S sigmaInv
//                    // innovation covariance
//                    sigma[0] = (P[0] * J[0] + J[2] * P[1]) * J[0] + (J[0] * P[2] + P[3] * J[2]) * J[2] + pow(dev_config.stdRange,2) ;
//                    sigma[1] = (P[0] * J[1] + J[3] * P[1]) * J[0] + (J[1] * P[2] + P[3] * J[3]) * J[2];
//                    sigma[2] = (P[0] * J[0] + J[2] * P[1]) * J[1] + (J[0] * P[2] + P[3] * J[2]) * J[3];
//                    sigma[3] = (P[0] * J[1] + J[3] * P[1]) * J[1] + (J[1] * P[2] + P[3] * J[3]) * J[3] + pow(dev_config.stdBearing,2) ;

//                    // enforce symmetry
//                    sigma[1] = (sigma[1]+sigma[2])/2 ;
//                    sigma[2] = sigma[1] ;
//        //			makePositiveDefinite(sigma) ;

//                    detSigma = sigma[0]*sigma[3] - sigma[1]*sigma[2] ;
//                    sigmaInv[0] = sigma[3]/detSigma ;
//                    sigmaInv[1] = -sigma[1]/detSigma ;
//                    sigmaInv[2] = -sigma[2]/detSigma ;
//                    sigmaInv[3] = sigma[0]/detSigma ;

//                    // Kalman gain
//                    K[0] = S[0]*(P[0]*J[0] + P[2]*J[2]) + S[1]*(P[0]*J[1] + P[2]*J[3]) ;
//                    K[1] = S[0]*(P[1]*J[0] + P[3]*J[2]) + S[1]*(P[1]*J[1] + P[3]*J[3]) ;
//                    K[2] = S[2]*(P[0]*J[0] + P[2]*J[2]) + S[3]*(P[0]*J[1] + P[2]*J[3]) ;
//                    K[3] = S[2]*(P[1]*J[0] + P[3]*J[2]) + S[3]*(P[1]*J[1] + P[3]*J[3]) ;

//                    // Updated covariance (Joseph Form)
//                    covUpdate[0] = ((1 - K[0] * J[0] - K[2] * J[1]) * P[0] + (-K[0] * J[2] - K[2] * J[3]) * P[1]) * (1 - K[0] * J[0] - K[2] * J[1]) + ((1 - K[0] * J[0] - K[2] * J[1]) * P[2] + (-K[0] * J[2] - K[2] * J[3]) * P[3]) * (-K[0] * J[2] - K[2] * J[3]) + pow(K[0], 2) * dev_config.stdRange*dev_config.stdRange + pow(K[2], 2) * dev_config.stdBearing*dev_config.stdBearing;
//                    covUpdate[2] = ((1 - K[0] * J[0] - K[2] * J[1]) * P[0] + (-K[0] * J[2] - K[2] * J[3]) * P[1]) * (-K[1] * J[0] - K[3] * J[1]) + ((1 - K[0] * J[0] - K[2] * J[1]) * P[2] + (-K[0] * J[2] - K[2] * J[3]) * P[3]) * (1 - K[1] * J[2] - K[3] * J[3]) + K[0] * dev_config.stdRange*dev_config.stdRange * K[1] + K[2] * dev_config.stdBearing*dev_config.stdBearing * K[3];
//                    covUpdate[1] = ((-K[1] * J[0] - K[3] * J[1]) * P[0] + (1 - K[1] * J[2] - K[3] * J[3]) * P[1]) * (1 - K[0] * J[0] - K[2] * J[1]) + ((-K[1] * J[0] - K[3] * J[1]) * P[2] + (1 - K[1] * J[2] - K[3] * J[3]) * P[3]) * (-K[0] * J[2] - K[2] * J[3]) + K[0] * dev_config.stdRange*dev_config.stdRange * K[1] + K[2] * dev_config.stdBearing*dev_config.stdBearing * K[3];
//                    covUpdate[3] = ((-K[1] * J[0] - K[3] * J[1]) * P[0] + (1 - K[1] * J[2] - K[3] * J[3]) * P[1]) * (-K[1] * J[0] - K[3] * J[1]) + ((-K[1] * J[0] - K[3] * J[1]) * P[2] + (1 - K[1] * J[2] - K[3] * J[3]) * P[3]) * (1 - K[1] * J[2] - K[3] * J[3]) + pow(K[1], 2) * dev_config.stdRange*dev_config.stdRange + pow(K[3], 2) * dev_config.stdBearing*dev_config.stdBearing;

//            #undef P
//            #undef S
        //	#undef J0
        //	#undef J1
        //	#undef J2
        //	#undef J3
                    /*
                     * END PRECOMPUTE UPDATE COMPONENTS
                     */

                    // save the non-detection term


                    int nonDetectIdx = update_offset + feature_idx ;
                    REAL nonDetectWeight = feature.weight * (1-featurePd) ;
                    updated_features[nonDetectIdx].weight = nonDetectWeight ;
                    updated_features[nonDetectIdx].mean[0] = feature.mean[0] ;
                    updated_features[nonDetectIdx].mean[1] = feature.mean[1] ;
                    updated_features[nonDetectIdx].cov[0] = feature.cov[0] ;
                    updated_features[nonDetectIdx].cov[1] = feature.cov[1] ;
                    updated_features[nonDetectIdx].cov[2] = feature.cov[2] ;
                    updated_features[nonDetectIdx].cov[3] = feature.cov[3] ;
    //				mergedFlags[nonDetectIdx] = false ;
                    if ( nonDetectWeight >= dev_config.minFeatureWeight )
                        mergedFlags[nonDetectIdx] = false ;
                    else
                        mergedFlags[nonDetectIdx] = true ;
                }
                else
                {
                    featurePd = 0 ;
                    feature.weight = 0 ;
                    covUpdate[0] = 0 ;
                    covUpdate[1] = 0 ;
                    covUpdate[2] = 0 ;
                    covUpdate[3] = 0 ;
                }

                // predicted cardinality = sum of weights*pd
                sumByReduction(sdata, feature.weight*featurePd, tid) ;
                if ( tid == 0 )
                    cardinality_predict += sdata[0] ;
                __syncthreads() ;

                /*
                 * LOOP THROUGH MEASUREMENTS AND DO UPDATE
                 */
                for (int i = 0 ; i < n_measure ; i++ )
                {
                    z = Z[i] ;
                    if ( feature_idx < n_features)
                    {
                        updateIdx = update_offset + (i+1)*n_features + feature_idx ;
                        // compute innovation
                        innov[0] = z.range - z_predict.range ;
                        innov[1] = wrapAngle( z.bearing - z_predict.bearing ) ;
                        // compute updated mean
                        meanUpdate[0] = feature.mean[0] + K[0]*innov[0] + K[2]*innov[1] ;
                        meanUpdate[1] = feature.mean[1] + K[1]*innov[0] + K[3]*innov[1] ;
                        // compute single object likelihood
                        dist = innov[0]*innov[0]*sigmaInv[0] +
                                innov[0]*innov[1]*(sigmaInv[1] + sigmaInv[2]) +
                                innov[1]*innov[1]*sigmaInv[3] ;

                        // Gating for births
                        // TODO: does this need to be atomic?
                        if ( dev_config.gateBirths && dist < dev_config.gateThreshold )
                            compatibleZ[i+map_idx*n_measure] |= 1 ;

                        // Gating for measurement update
                        if ( dev_config.gateMeasurements && dist > dev_config.gateThreshold)
                            dist = FLT_MAX ;


                        // partially updated weight
                        if ( featurePd > 0 )
                        {
                            logLikelihood = safeLog(featurePd) + safeLog(feature.weight)
                                    - 0.5*dist -  0.5*safeLog(2*M_PI*detSigma) ;
                            w_partial = logLikelihood ;
                        }
                        else
                        {
                            w_partial = 0 ;
                        }
                        // save updated gaussian with partially updated weight
                        updated_features[updateIdx].weight = w_partial ;
                        updated_features[updateIdx].mean[0] = meanUpdate[0] ;
                        updated_features[updateIdx].mean[1] = meanUpdate[1] ;
                        updated_features[updateIdx].cov[0] = covUpdate[0] ;
                        updated_features[updateIdx].cov[1] = covUpdate[1] ;
                        updated_features[updateIdx].cov[2] = covUpdate[2] ;
                        updated_features[updateIdx].cov[3] = covUpdate[3] ;
                    }
                }
            }

            // choose the feature for the single-feature particle weight update
            if ( dev_config.particleWeighting == 2 )
            {
                REAL max_weight = -FLT_MAX ;
                for ( int n = 0 ; n < n_features ; n += blockDim.x )
                {
                    sdata[tid] = -FLT_MAX ;
                    __syncthreads() ;
                    if ( tid+n < n_features)
                    {
                        for ( int i = 0 ;i < n_measure ; i += 1 )
                        {
                            int idx = update_offset + (i+1)*n_features + tid + n ;
                            sdata[tid] = fmax(sdata[tid],updated_features[idx].weight ) ;
                        }
                    }
                    __syncthreads() ;
                    if ( tid == 0 )
                    {
                        for ( int i = 0 ; i < blockDim.x ; i++ )
                        {
                            if (sdata[i] > max_weight )
                            {
                                max_feature = n+i ;
                                max_weight = sdata[i] ;
                            }
                        }

                    }
                    __syncthreads() ;
                }
                REAL tmp = 0 ;
                if ( tid < n_measure )
                {
                    int idx = update_offset + n_features
                            + tid*n_features + max_feature ;
                    tmp = exp(updated_features[idx].weight) ;
                }
                sumByReduction(sdata,tmp,tid) ;
                __syncthreads() ;
                if ( tid == 0 )
                {
                    particle_weight = updated_features[update_offset+max_feature].weight*pow(dev_config.clutterDensity,n_measure)
                            + pow(dev_config.clutterDensity,n_measure-1)*sdata[0] ;
                }
                __syncthreads() ;
            }

            // compute the weight normalizers
            for ( int i = 0 ; i < n_measure ; i++ )
            {
                REAL log_normalizer = 0 ;
                REAL val = -FLT_MAX ;
                // find the maximum from all the log partial weights
                for ( int j = 0 ; j < n_features ; j += blockDim.x )
                {
                    int feature_idx = j+tid ;
                    if ( feature_idx < n_features )
                    {
                        updateIdx = update_offset + (i+1)*n_features + feature_idx ;
                        w_partial = updated_features[updateIdx].weight ;
                    }
                    else
                    {
                        w_partial = -FLT_MAX ;
                    }
                    maxByReduction( sdata, w_partial, tid ) ;
                    val = fmax(val,sdata[0]) ;
                    __syncthreads() ;
                }

                // do the exponent sum
                REAL sum = 0 ;
                for ( int j = 0 ; j < n_features ; j += blockDim.x )
                {
                    int feature_idx = j+tid ;
                    if ( feature_idx < n_features )
                    {
                        updateIdx = update_offset + (i+1)*n_features + feature_idx ;
                        w_partial = exp(updated_features[updateIdx].weight-val) ;
                    }
                    else
                    {
                        w_partial = 0 ;
                    }
                    sumByReduction( sdata, w_partial, tid ) ;
                    sum += sdata[0] ;
                    __syncthreads() ;
                }
                sum = exp(safeLog(sum) + val) ;
                sum += dev_config.clutterDensity ;

                // add back the offset value
                log_normalizer = safeLog(sum) ;

                if ( tid == 0 && dev_config.particleWeighting == 0 )
                    particle_weight += log_normalizer ;
                __syncthreads() ;
                for ( int j = 0 ; j < n_features ; j += blockDim.x )
                {
                    int feature_idx = j + tid ;
                    if ( feature_idx < n_features )
                    {
                        updateIdx = update_offset + (i+1)*n_features + feature_idx ;
                        weightUpdate = exp(updated_features[updateIdx].weight - log_normalizer) ;
                        updated_features[updateIdx].weight = weightUpdate ;
                        if ( weightUpdate >= dev_config.minFeatureWeight)
                        {
                            mergedFlags[updateIdx] = false ;
                        }
                        else
                        {
                            mergedFlags[updateIdx] = true ;
                        }
                    }
                }
            }

            // Particle weighting
            __syncthreads() ;
            // Cluster-PHD particle weighting
            if ( dev_config.particleWeighting == 0 )
            {
                if ( tid == 0 )
                {
                    particle_weight -= cardinality_predict ;
                    particleWeights[map_idx] = particle_weight ;
                }
            }
            // Vo-EmptyMap particle weighting
            else if ( dev_config.particleWeighting == 1 )
            {
                REAL cn_predict = 0 ;
                REAL cn_update = 0 ;

                // predicted cardinality
                for ( int i = 0 ; i < n_features ; i+= blockDim.x)
                {
                    if ( tid + i < n_features )
                        w_partial = inRangeFeatures[predict_offset+i+tid].weight ;
                    else
                        w_partial = 0.0f ;
                    sumByReduction(sdata,w_partial,tid);
                    cn_predict += sdata[0] ;
                    __syncthreads() ;
                }

                // updated cardinality = sum of updated weights
                for ( int i = 0 ; i < n_update ; i += blockDim.x )
                {
                    // to avoid divergence in the reduction function call, load weight
                    // into temp variable first
                    if ( tid + i < n_update )
                        w_partial = updated_features[update_offset+i+tid].weight ;
                    else
                        w_partial = 0.0f ;
                    sumByReduction( sdata, w_partial, tid );
                    if ( tid == 0 )
                        cn_update += sdata[0] ;
                    __syncthreads() ;
                }
                // thread 0 computes the weight
                if ( tid == 0 )
                {
                    particleWeights[map_idx] = n_measure*safeLog(dev_config.clutterDensity)
                            + cn_update - cn_predict
                            - dev_config.clutterRate  ;
//                    particleWeights[map_idx] = cardinality_updated
//                            - cardinality_predict ;
                }
            }
            // Single-Feature IID cluster assumption
            else if ( dev_config.particleWeighting == 2 )
            {
//                REAL cn_predict = 0 ;
//                REAL cn_update = 0 ;

//                // predicted cardinality
//                for ( int i = 0 ; i < n_features ; i+= blockDim.x)
//                {
//                    if ( tid + i < n_features )
//                        w_partial = inRangeFeatures[predict_offset+i+tid].weight ;
//                    else
//                        w_partial = 0.0f ;
//                    sumByReduction(sdata,w_partial,tid);
//                    cn_predict += sdata[0] ;
//                    __syncthreads() ;
//                }

//                // updated cardinality = sum of updated weights
//                for ( int i = 0 ; i < n_update ; i += blockDim.x )
//                {
//                    if ( tid + i < n_update )
//                        w_partial = updated_features[update_offset+i+tid].weight ;
//                    else
//                        w_partial = 0.0f ;
//                    sumByReduction( sdata, w_partial, tid );
//                    if ( tid == 0 )
//                        cn_update += sdata[0] ;
//                    __syncthreads() ;
//                }

                if ( tid == 0 )
                {
                    w_partial = 0 ;
                    for ( int i = 0 ; i <= n_measure ; i++ )
                        w_partial += updated_features[update_offset + i*n_features + max_feature].weight ;
                    REAL log_den = safeLog(w_partial) ;
//                    REAL log_den = safeLog(w_partial)
//                            + cn_predict - cn_update + dev_config.clutterRate ;
//                    REAL log_den = n_measure*safeLog(dev_config.clutterDensity)
//                            + safeLog(updated_features[update_offset+max_feature].weight) ;
                    particleWeights[map_idx] = safeLog(particle_weight) - log_den ;
                }
            }
        }
    }
}

__global__ void
phdUpdateMergeKernel(Gaussian2D *updated_features,
                     Gaussian2D *mergedFeatures, int *mergedSizes,
                     bool *mergedFlags, int* mapSizes, int nParticles )
{
    __shared__ Gaussian2D maxFeature ;
    __shared__ REAL wMerge ;
    __shared__ REAL meanMerge[2] ;
    __shared__ REAL covMerge[4] ;
    __shared__ REAL sdata[256] ;
    __shared__ int mergedSize ;
    __shared__ int update_offset ;
    __shared__ int n_update ;
    int tid = threadIdx.x ;
    REAL dist ;
    REAL innov[2] ;
    Gaussian2D feature ;
    // loop over particles
    for ( int p = 0 ; p < nParticles ; p += gridDim.x )
    {
        int map_idx = p + blockIdx.x ;
        if ( map_idx <= nParticles )
        {
            // initialize shared vars
            if ( tid == 0)
            {
                update_offset = 0 ;
                for ( int i = 0 ; i < map_idx ; i++ )
                {
                    update_offset += mapSizes[i] ;
                }
                n_update = mapSizes[map_idx] ;
                mergedSize = 0 ;
            }
            __syncthreads() ;
            while(true)
            {
                // initialize the output values to defaults
                if ( tid == 0 )
                {
                    maxFeature.weight = -1 ;
                    wMerge = 0 ;
                    meanMerge[0] = 0 ;
                    meanMerge[1] = 0 ;
                    covMerge[0] = 0 ;
                    covMerge[1] = 0 ;
                    covMerge[2] = 0 ;
                    covMerge[3] = 0 ;
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
                {
                    break ;
                }
                else if(tid == 0)
                {
                    maxFeature = updated_features[ (unsigned int)sdata[0] ] ;
//                    wMerge = maxFeature.weight ;
//                    meanMerge[0] = maxFeature.mean[0] ;
//                    meanMerge[1] = maxFeature.mean[1] ;
//                    covMerge[0] = maxFeature.cov[0] ;
//                    covMerge[1] = maxFeature.cov[1] ;
//                    covMerge[2] = maxFeature.cov[2] ;
//                    covMerge[3] = maxFeature.cov[3] ;
//                    mergedFlags[ (unsigned int)sdata[0] ] = true ;
                }
                __syncthreads() ;

                // find features to merge with max feature
                REAL sval0 = 0 ;
                REAL sval1 = 0 ;
                REAL sval2 = 0 ;
                for ( int i = update_offset ; i < update_offset + n_update ; i += blockDim.x )
                {
                    int idx = tid + i ;
                    if ( idx < update_offset+n_update )
                    {
                        if ( !mergedFlags[idx] )
                        {
                            feature = updated_features[idx] ;
                            if ( dev_config.distanceMetric == 0 )
                                dist = computeMahalDist(maxFeature, feature) ;
                            else if ( dev_config.distanceMetric == 1)
                                dist = computeHellingerDist(maxFeature, feature) ;
                            if ( dist < dev_config.minSeparation )
                            {
                                sval0 += feature.weight ;
                                sval1 += feature.mean[0]*feature.weight ;
                                sval2 += feature.mean[1]*feature.weight ;
                            }
                        }
                    }
                }
                sumByReduction(sdata, sval0, tid) ;
                if ( tid == 0 )
                    wMerge += sdata[0] ;
                __syncthreads() ;
                if ( wMerge == 0 )
                    break ;
                sumByReduction( sdata, sval1, tid ) ;
                if ( tid == 0 )
                    meanMerge[0] += sdata[0]/wMerge ;
                __syncthreads() ;
                sumByReduction( sdata, sval2, tid ) ;
                if ( tid == 0 )
                    meanMerge[1] += sdata[0]/wMerge ;
                __syncthreads() ;


                // merge the covariances
                sval0 = 0 ;
                sval1 = 0 ;
                sval2 = 0 ;
                for ( int i = update_offset ; i < update_offset+n_update ; i += blockDim.x )
                {
                    int idx = tid + i ;
                    if ( idx < update_offset+n_update )
                    {
                        if (!mergedFlags[idx])
                        {
                            feature = updated_features[idx] ;
                            if ( dev_config.distanceMetric == 0 )
                                dist = computeMahalDist(maxFeature, feature) ;
                            else if ( dev_config.distanceMetric == 1)
                                dist = computeHellingerDist(maxFeature, feature) ;
                            if ( dist < dev_config.minSeparation )
                            {
                                innov[0] = meanMerge[0] - feature.mean[0] ;
                                innov[1] = meanMerge[1] - feature.mean[1] ;
                                sval0 += (feature.cov[0] + innov[0]*innov[0])*feature.weight ;
                                sval1 += (feature.cov[1] +feature.cov[2] + 2*innov[0]*innov[1])*feature.weight/2 ;
                                sval2 += (feature.cov[3] + innov[1]*innov[1])*feature.weight ;
                                mergedFlags[idx] = true ;
                            }
                        }
                    }
                }
                sumByReduction(sdata, sval0, tid) ;
                if ( tid == 0 )
                    covMerge[0] += sdata[0]/wMerge ;
                __syncthreads() ;
                sumByReduction( sdata, sval1, tid ) ;
                if ( tid == 0 )
                {
                    covMerge[1] += sdata[0]/wMerge ;
                    covMerge[2] = covMerge[1] ;
                }
                __syncthreads() ;
                sumByReduction( sdata, sval2, tid ) ;
                if ( tid == 0 )
                {
                    covMerge[3] += sdata[0]/wMerge ;
                    int mergeIdx = update_offset + mergedSize ;
                    mergedFeatures[mergeIdx].weight = wMerge ;
                    mergedFeatures[mergeIdx].mean[0] = meanMerge[0] ;
                    mergedFeatures[mergeIdx].mean[1] = meanMerge[1] ;
                    mergedFeatures[mergeIdx].cov[0] = covMerge[0] ;
                    mergedFeatures[mergeIdx].cov[1] = covMerge[1] ;
                    mergedFeatures[mergeIdx].cov[2] = covMerge[2] ;
                    mergedFeatures[mergeIdx].cov[3] = covMerge[3] ;
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


//__global__ void
//fastSlamUpdateKernel( Gaussian2D* predict_features, ConstantVelocityState* poses,
//                      int* map_offsets, int n_measure, REAL* log_g )
//{
//    int f_idx = threadIdx.x ;
//    int z_idx = threadIdx.y ;
//    int map_idx = blockIdx.x ;
//    int n_features = map_offsets[map_idx+1] - map_offsets[map_idx] ;
//    int predict_offset = map_offsets[map_idx] ;
//    int update_offset = predict_offset*(n_measure) ;
//    int g_idx = update_offset + z_idx*n_features + f_idx ;

//    Gaussian2D feature = predict_features[predict_offset+f_idx] ;
//    ConstantVelocityState pose = poses[map_idx] ;

//    RangeBearingMeasurement z_predict ;
//    REAL K[4] ;
//    REAL S_inverse[4] ;
//    REAL cov_update[4] ;
//    REAL det_sigma ;
//    REAL feature_pd ;
//    REAL S_inv[4] ;
//    computePreUpdateComponents(pose,feature,K,cov_update,&det_sigma,S_inv,
//                               &feature_pd,&z_predict) ;
//}

//void
//fastSlamDataAssociation()
//{
//    // compute individual compatibility
//    // compute jcbb association
//}

ParticleSLAM
phdUpdate(ParticleSLAM& particles, measurementSet measurements)
{
    // make a copy of the particles
    ParticleSLAM particlesPreMerge(particles) ;

    ///////////////////////////////////////////////////////////////////////////
    //
    // concatenate all the maps together for parallel processing
    //
    ///////////////////////////////////////////////////////////////////////////
    gaussianMixture concat ;
    int nParticles = particles.nParticles ;
    int n_measure = measurements.size() ;
    vector<int> mapSizes(nParticles) ;
//	vector<REAL> concat_cn ;
    int nThreads = 0 ;
    int totalFeatures = 0 ;
    DEBUG_VAL(nParticles) ;
    for ( unsigned int n = 0 ; n < nParticles ; n++ )
    {
        concat.insert( concat.end(), particles.maps[n].begin(),
                particles.maps[n].end()) ;
        // to debug the cphd, generate a poisson cardinality
        REAL weight_sum = 0 ;
        for ( int j = 0 ; j < particles.maps[n].size() ; j++ )
            weight_sum += particles.maps[n][j].weight ;
        mapSizes[n] = particles.maps[n].size() ;

        // keep track of largest map feature count
        if ( mapSizes[n] > nThreads )
            nThreads = mapSizes[n] ;
        nThreads = min(nThreads,256) ;
        totalFeatures += mapSizes[n] ;
    }
    if ( totalFeatures == 0 )
    {
        DEBUG_MSG("no features, exiting early") ;
        return particlesPreMerge;
    }

    ///////////////////////////////////////////////////////////////////////////
    //
    // split features into in/out range parts
    //
    ///////////////////////////////////////////////////////////////////////////

    // allocate device memory
    Gaussian2D* dev_maps = NULL ;
    int* dev_map_sizes = NULL ;
    int* dev_n_in_range = NULL ;
    char* dev_in_range = NULL ;
    ConstantVelocityState* dev_poses = NULL ;
    CUDA_SAFE_CALL(
                cudaMalloc( (void**)&dev_maps,
                            totalFeatures*sizeof(Gaussian2D) ) ) ;
    CUDA_SAFE_CALL(
                cudaMalloc( (void**)&dev_map_sizes,
                            nParticles*sizeof(int) ) ) ;
    CUDA_SAFE_CALL(
                cudaMalloc( (void**)&dev_n_in_range,
                            nParticles*sizeof(int) ) ) ;
    CUDA_SAFE_CALL(
                cudaMalloc( (void**)&dev_in_range,
                            totalFeatures*sizeof(char) ) ) ;
    CUDA_SAFE_CALL(
            cudaMalloc( (void**)&dev_poses,
                        nParticles*sizeof(ConstantVelocityState) ) ) ;

    // copy inputs
    CUDA_SAFE_CALL(
        cudaMemcpy( dev_maps, &concat[0], totalFeatures*sizeof(Gaussian2D),
                    cudaMemcpyHostToDevice )
    ) ;
    CUDA_SAFE_CALL(
                cudaMemcpy( dev_map_sizes, &mapSizes[0], nParticles*sizeof(int),
                    cudaMemcpyHostToDevice )
    ) ;
    CUDA_SAFE_CALL(
                cudaMemcpy(dev_poses,&particles.states[0],
                           nParticles*sizeof(ConstantVelocityState),
                           cudaMemcpyHostToDevice)
    ) ;

    // kernel launch
    DEBUG_MSG("launching computeInRangeKernel") ;
    DEBUG_VAL(nThreads) ;
    computeInRangeKernel<<<nParticles,nThreads>>>
        ( dev_maps, dev_map_sizes, nParticles, dev_poses, dev_in_range, dev_n_in_range) ;
    CUDA_SAFE_THREAD_SYNC();

    // allocate outputs
    vector<char> in_range(totalFeatures) ;
    vector<int> n_in_range_vec(nParticles) ;

    // copy outputs
    CUDA_SAFE_CALL(
                cudaMemcpy( &in_range[0],dev_in_range,totalFeatures*sizeof(char),
                    cudaMemcpyDeviceToHost )
    ) ;
    CUDA_SAFE_CALL(
        cudaMemcpy( &n_in_range_vec[0],dev_n_in_range,nParticles*sizeof(int),
                    cudaMemcpyDeviceToHost )
    ) ;

    // get total number of in-range features
    int n_in_range = 0 ;
    vector<int> n_out_range_vec(nParticles) ;
    for ( int i = 0 ; i < nParticles ; i++ )
    {
        n_in_range += n_in_range_vec[i] ;
        n_out_range_vec[i] = particles.maps[i].size() -  n_in_range_vec[i] ;
    }
    int n_out_range = totalFeatures - n_in_range ;

    // divide features into in-range/out-of-range parts
    DEBUG_VAL(n_in_range) ;
    DEBUG_VAL(n_out_range) ;
    gaussianMixture features_in(n_in_range) ;
    gaussianMixture features_out(n_out_range ) ;
    int idx_in = 0 ;
    int idx_out = 0 ;
    for ( int i = 0 ; i < totalFeatures ; i++ )
    {
        if (in_range[i] == 1)
            features_in[idx_in++] = concat[i] ;
        else
            features_out[idx_out++] = concat[i] ;
    }

    // free memory
    CUDA_SAFE_CALL( cudaFree( dev_maps ) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_in_range ) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_n_in_range ) ) ;

    ///////////////////////////////////////////////////////////////////////////
    //
    // Do PHD update with only in-range features
    //
    ///////////////////////////////////////////////////////////////////////////

//	if ( config.filterType == CPHD_TYPE )
//	{
//		n_in_range_vec = mapSizes ;
//		n_in_range = totalFeatures ;
//		n_out_range = 0 ;
//		features_in = concat ;

//	}


    // check for memory limit for storing measurements in constant mem
    if ( n_measure > 256 )
    {
        DEBUG_MSG("Warning: maximum number of measurements per time step exceeded") ;
//		DEBUG_VAL(n_measure-MAXMEASUREMENTS) ;
        n_measure = 256 ;
    }
    DEBUG_VAL(n_measure) ;

    int n_update = n_in_range*(n_measure+1) ;

    // perform an (inclusive) prefix scan on the map sizes to determine indexing
    // offsets for each map
    vector<int> map_offsets_in(nParticles+1,0) ;
    vector<int> map_offsets_out(nParticles+1,0) ;
    for ( int i = 1 ; i < nParticles+1 ; i++ )
    {
        map_offsets_in[i] = map_offsets_in[i-1] + n_in_range_vec[i-1] ;
        map_offsets_out[i] = map_offsets_out[i-1] + n_in_range_vec[i-1] ;
    }

    // allocate device memory
    int *dev_map_sizes_inrange = NULL ;
    int *dev_map_offsets_inrange = NULL ;
    int* dev_n_merged = NULL ;
    char* dev_compatible_z = NULL ;
    Gaussian2D* dev_maps_inrange = NULL ;
    Gaussian2D* dev_maps_updated = NULL ;
    Gaussian2D* dev_maps_merged = NULL ;
    REAL* dev_particle_weights = NULL ;
    bool* dev_merged_flags = NULL ;

    // cphd-only variables
    REAL* dev_wpartial = NULL ;
    REAL* dev_esf = NULL ;
    REAL* dev_esfd = NULL ;
    REAL* dev_innerprod0 = NULL ;
    REAL* dev_innerprod1 = NULL ;
    REAL* dev_innerprod1d = NULL ;
    REAL* dev_cnpredict = NULL ;
    REAL* dev_cnupdate = NULL ;
    REAL* dev_qdw = NULL ;

    CUDA_SAFE_CALL(
                    cudaMalloc( (void**)&dev_maps_inrange,
                                n_in_range*sizeof(Gaussian2D) ) ) ;
    CUDA_SAFE_CALL(
                    cudaMalloc( (void**)&dev_map_sizes_inrange,
                                nParticles*sizeof(int) ) ) ;
    CUDA_SAFE_CALL(
                    cudaMalloc( (void**)&dev_map_offsets_inrange,
                                (nParticles+1)*sizeof(int) ) ) ;
    CUDA_SAFE_CALL(
                cudaMalloc((void**)&dev_compatible_z,
                           nParticles*n_measure*sizeof(char) ) ) ;
    CUDA_SAFE_CALL(
                cudaMalloc((void**)&dev_maps_updated,
                           n_update*sizeof(Gaussian2D)) ) ;
    CUDA_SAFE_CALL(
                cudaMalloc((void**)&dev_particle_weights,
                           nParticles*sizeof(REAL) ) ) ;
    CUDA_SAFE_CALL(
                cudaMalloc((void**)&dev_n_merged,
                           nParticles*sizeof(int)) ) ;
    CUDA_SAFE_CALL(
                cudaMalloc((void**)&dev_merged_flags,
                           n_update*sizeof(bool)) ) ;
    if ( config.filterType == CPHD_TYPE )
    {
        size_t cn_size = nParticles*(config.maxCardinality+1)*sizeof(REAL) ;
        CUDA_SAFE_CALL(
                    cudaMalloc( (void**)&dev_cnpredict, cn_size ) ) ;
        CUDA_SAFE_CALL(
                    cudaMalloc( (void**)&dev_cnupdate, cn_size ) ) ;
        CUDA_SAFE_CALL(
                    cudaMalloc( (void**)&dev_wpartial,
                                n_measure*totalFeatures*sizeof(REAL) ) ) ;
        CUDA_SAFE_CALL(
                    cudaMalloc( (void**)&dev_esf,
                                (n_measure+1)*nParticles*sizeof(REAL) ) ) ;
        CUDA_SAFE_CALL(
                    cudaMalloc( (void**)&dev_esfd,
                                n_measure*n_measure*nParticles*sizeof(REAL) ) ) ;
        CUDA_SAFE_CALL(
                    cudaMalloc( (void**)&dev_innerprod0,
                                nParticles*sizeof(REAL) ) ) ;
        CUDA_SAFE_CALL(
                    cudaMalloc( (void**)&dev_innerprod1,
                                nParticles*sizeof(REAL) ) ) ;
        CUDA_SAFE_CALL(
                    cudaMalloc( (void**)&dev_innerprod1d,
                                nParticles*n_measure*sizeof(REAL) ) ) ;
        CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_qdw,
                                    totalFeatures*sizeof(REAL) ) ) ;
    }

    // copy inputs
    CUDA_SAFE_CALL(
        cudaMemcpy( dev_maps_inrange, &features_in[0],
                    n_in_range*sizeof(Gaussian2D),
                    cudaMemcpyHostToDevice )
    ) ;
    CUDA_SAFE_CALL(
        cudaMemcpy( dev_map_sizes_inrange, &n_in_range_vec[0],
                    nParticles*sizeof(int),
                    cudaMemcpyHostToDevice )
        ) ;
    CUDA_SAFE_CALL( cudaMemcpy( dev_map_offsets_inrange, &map_offsets_in[0],
                                (nParticles+1)*sizeof(int),
                                cudaMemcpyHostToDevice ) ) ;
    CUDA_SAFE_CALL(
                cudaMemcpyToSymbol( Z, &measurements[0],
                                    n_measure*sizeof(RangeBearingMeasurement) ) ) ;

    // compute the in-range cardinality for the CPHD
    if ( config.filterType == CPHD_TYPE )
    {
        int offset = 0 ;
//        int cn_offset = 0 ;
        size_t cn_size = nParticles*(config.maxCardinality+1)*sizeof(REAL) ;
//        REAL* cn_predict = (REAL*)malloc(cn_size) ;
        vector<REAL> cn_predict ;

        // binomial poisson cardinality
        for ( int i = 0 ; i < nParticles ; i++ )
        {
            vector<REAL> cn_predict_i(config.maxCardinality+1) ;
            std::fill( cn_predict_i.begin(),cn_predict_i.end(),0) ;
            cn_predict_i[0] = 1 ;
            for ( int j = 0 ; j < n_in_range_vec[i] ; j++ )
            {
                vector<REAL> b(2) ;
                b[0] = 1-features_in[offset+j].weight ;
                b[1] = features_in[offset+j].weight ;
                cn_predict_i = conv(cn_predict_i,b) ;
            }
            for ( int n = 0 ; n <= config.maxCardinality ; n++ )
            {
                cn_predict_i[n] = safeLog(cn_predict_i[n]) ;
            }
            cn_predict.insert( cn_predict.end(),cn_predict_i.begin(),
                               cn_predict_i.begin()+config.maxCardinality+1 ) ;
            offset += n_in_range_vec[i] ;
        }
//        for ( int i = 0 ; i < nParticles ; i++ )
//        {
//                int n_fft = n_in_range_vec[i]+1 ;
//                fftw_complex* in = (fftw_complex*)fftw_malloc(n_fft*sizeof(fftw_complex) ) ;
//                fftw_complex* out = (fftw_complex*)fftw_malloc(n_fft*sizeof(fftw_complex) ) ;
//                fftw_plan p = fftw_plan_dft_1d(n_fft,in,out,FFTW_FORWARD,FFTW_ESTIMATE) ;
//                for ( int j = 0 ; j < n_fft ; j++ )
//                {
//                        in[j] = 1 + 0*I ;
//                        for ( int k = 0 ; k < n_in_range_vec[i] ; k++ )
//                        {
//                                in[j] *= (cexp( I*2*M_PI*j/n_fft ) - 1)*features_in[offset+k].weight
//                                                + 1 ;
//                        }
//                }
//                fftw_execute(p) ;
//                for ( int n = 0 ; n <= config.maxCardinality ; n++ )
//                {
//                        if ( n < n_fft )
//                                cn_predict[cn_offset+n] = safeLog(cabs(out[n]))-safeLog(n_fft) ;
//                        else
//                                cn_predict[cn_offset+n] = LOG0 ;
//                }
//                fftw_destroy_plan(p) ;
//                fftw_free(in) ;
//                fftw_free(out) ;
//                offset += n_in_range_vec[i] ;
//                cn_offset += config.maxCardinality+1 ;
//        }


//        // poisson cardinality
//        vector<REAL> log_factorials( config.maxCardinality+1) ;
//        log_factorials[0] = 0 ;
//        for ( int n = 1 ; n <= config.maxCardinality ; n++ )
//        {
//            log_factorials[n] = log_factorials[n-1] + safeLog((REAL)n) ;
//        }
//        for ( int n = 0 ; n < nParticles ; n++ )
//        {
//            REAL w_sum = 0 ;
//            for ( int i = 0 ; i < particles.maps[n].size() ; i++ )
//                w_sum += particles.maps[n][i].weight ;
//            for ( int i = 0 ; i < config.maxCardinality+1 ; i++ )
//            {
//                cn_predict[offset++] = i*safeLog(w_sum)
//                        - w_sum
//                        - log_factorials[i] ;
//            }
////			for ( int i = 0 ; i < (config.maxCardinality+1) ; i++ )
////			{
////				if ( i==round(w_sum) )
////					cn_predict[offset++] = 0 ;
////				else
////					cn_predict[offset++] = LOG0 ;
////			}
//        }

        CUDA_SAFE_CALL( cudaMemcpy(dev_cnpredict,&cn_predict[0],cn_size,
                                   cudaMemcpyHostToDevice) ) ;
//        free(cn_predict) ;
    }


    // launch kernel
    int nBlocks = min(nParticles,32768) ;

    if ( config.filterType == PHD_TYPE )
    {
        DEBUG_MSG("launching phdUpdateKernel") ;
        phdUpdateKernel<<<nBlocks,256>>>
            ( dev_maps_inrange, dev_map_offsets_inrange, nParticles, n_measure, dev_poses,
              dev_compatible_z, dev_maps_updated,
              dev_merged_flags,dev_particle_weights ) ;
        CUDA_SAFE_THREAD_SYNC() ;
    }
    else if ( config.filterType == CPHD_TYPE )
    {
        int n_blocks = ceil( (float)n_update/128 ) ;
        cphdPreUpdateKernel<<<n_blocks, 128>>>
            ( dev_maps_inrange,	dev_map_offsets_inrange,nParticles,
              n_measure,dev_poses, dev_maps_updated,dev_wpartial, dev_qdw ) ;
        CUDA_SAFE_THREAD_SYNC() ;

        size_t shmem_size = sizeof(REAL)*(2*n_measure + 1 ) ;
        computeEsfKernel<<<nParticles, n_measure,shmem_size>>>
            ( dev_wpartial, dev_map_offsets_inrange, n_measure, dev_esf,
              dev_esfd ) ;
        CUDA_SAFE_THREAD_SYNC() ;

        shmem_size = sizeof(REAL)*(config.maxCardinality+1) ;
//		shmem_size = 0 ;
        computePsiKernel<<<nParticles, config.maxCardinality+1, shmem_size>>>
            ( dev_maps_inrange, dev_cnpredict, dev_esf, dev_esfd,
              dev_map_offsets_inrange, n_measure, dev_qdw,
              dev_factorial, dev_C, dev_cn_clutter,
              dev_cnupdate, dev_innerprod0, dev_innerprod1, dev_innerprod1d ) ;
        CUDA_SAFE_THREAD_SYNC() ;

        cphdUpdateKernel<<<nParticles, n_measure>>>
            ( dev_map_offsets_inrange, n_measure,
              dev_innerprod0, dev_innerprod1, dev_innerprod1d, dev_merged_flags,
              dev_maps_updated ) ;
        CUDA_SAFE_THREAD_SYNC() ;
        CUDA_SAFE_CALL( cudaFree( dev_wpartial) ) ;
        CUDA_SAFE_CALL( cudaFree( dev_cnpredict) ) ;
        CUDA_SAFE_CALL( cudaFree( dev_qdw) ) ;
        CUDA_SAFE_CALL( cudaFree( dev_esf) ) ;
        CUDA_SAFE_CALL( cudaFree( dev_esfd) ) ;
        CUDA_SAFE_CALL( cudaFree( dev_innerprod1) ) ;
        CUDA_SAFE_CALL( cudaFree( dev_innerprod1d) ) ;
    }
    CUDA_SAFE_CALL( cudaFree( dev_maps_inrange ) ) ;

    /**********************************************************
      *
      * Update particle weights
      *
      *********************************************************/
    REAL* particle_weights = (REAL*)malloc(nParticles*sizeof(REAL)) ;
    REAL* cn_update = NULL ;
    if ( config.filterType == PHD_TYPE )
    {
        CUDA_SAFE_CALL(
                    cudaMemcpy(particle_weights,dev_particle_weights,
                               nParticles*sizeof(REAL),
                               cudaMemcpyDeviceToHost ) ) ;
//        // compute Vo empty map particle weighting
//        if ( config.particleWeighting==1 || config.particleWeighting==2)
//        {
//            Gaussian2D* maps_combined = (Gaussian2D*)malloc( combined_size ) ;
//            CUDA_SAFE_CALL(cudaMemcpy(maps_combined,dev_maps_combined,combined_size,
//                                      cudaMemcpyDeviceToHost) ) ;
//            offset_updated = 0 ;
//            for ( int i = 0 ; i < nParticles ; i++ )
//            {
//                // predicted cardinality
//                double cardinality_predict = 0 ;
//                gaussianMixture map_predict = particles.maps[i] ;
//                for ( int j = 0 ; j < map_predict.size() ; j++ )
//                    cardinality_predict += map_predict[j].weight ;

//                // updated cardinality
//                double cardinality_update = 0 ;
//                for ( int j = 0 ; j < mapSizes[i] ; j++)
//                    cardinality_update += maps_combined[offset_updated++].weight ;

//                if ( config.particleWeighting == 1)
//                {
//                    // compute particle weight
//                    particle_weights[i] = n_measure*safeLog(config.clutterDensity)
//                            + cardinality_update - cardinality_predict
//                            - config.clutterRate ;
//                }
//                else
//                {
//                    particle_weights[i] -= (cardinality_predict-cardinality_update+config.clutterDensity) ;
//                }
//            }
//            free(maps_combined) ;
//        }
    }
    else if ( config.filterType == CPHD_TYPE )
    {
        size_t cn_size = (config.maxCardinality+1)*nParticles*sizeof(REAL) ;
        cn_update = (REAL*)malloc( cn_size ) ;
        CUDA_SAFE_CALL(
                    cudaMemcpy(particle_weights,dev_innerprod0,
                               nParticles*sizeof(REAL),
                               cudaMemcpyDeviceToHost ) ) ;
        CUDA_SAFE_CALL( cudaMemcpy( cn_update, dev_cnupdate, cn_size,
                                    cudaMemcpyDeviceToHost ) ) ;
        CUDA_SAFE_CALL( cudaFree( dev_innerprod0) ) ;
        CUDA_SAFE_CALL( cudaFree( dev_cnupdate) ) ;
    }
    CUDA_SAFE_CALL( cudaFree( dev_particle_weights ) ) ;


    /***************************************************
      *
      * Merge updated Gaussian Mixtures
      *
      ***************************************************/
//    // only merge in-range maps
//    for (int n = 0 ; n < nParticles ; n++ )
//    {
//        mapSizes[n] = n_in_range_vec[n]*(n_measure+1) ;
//    }
//    CUDA_SAFE_CALL( cudaMemcpy( dev_map_sizes, &mapSizes[0],
//                                nParticles*sizeof(int),
//                                cudaMemcpyHostToDevice ) ) ;
//    phdUpdateMergeKernel<<<nBlocks,256>>>(dev_maps_updated, dev_maps_merged,
//                                          dev_n_merged, dev_merged_flags,
//                                          dev_map_sizes,nParticles);

    // recombine updated in-range map with out-of-range map and do merging
    Gaussian2D* dev_maps_combined = NULL ;
    bool* dev_merged_flags_combined = NULL ;
    size_t combined_size = (n_update+n_out_range)*sizeof(Gaussian2D) ;
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_maps_combined, combined_size ) ) ;
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_merged_flags_combined,
                                (n_update+n_out_range)*sizeof(bool) ) ) ;
    CUDA_SAFE_CALL( cudaMalloc((void**)&dev_maps_merged,
                           combined_size ) ) ;
    int offset = 0 ;
    int offset_updated = 0 ;
    int offset_out = 0 ;
    for ( int n = 0 ; n < nParticles ; n++ )
    {
        // out-of-range map for particle n
        vector<char> merged_flags_out(n_out_range_vec[n],0) ;
        CUDA_SAFE_CALL( cudaMemcpy( dev_maps_combined+offset,
                                    &features_out[offset_out],
                                    n_out_range_vec[n]*sizeof(Gaussian2D),
                                    cudaMemcpyHostToDevice ) ) ;
        CUDA_SAFE_CALL( cudaMemcpy( dev_merged_flags_combined+offset,
                                    &merged_flags_out[0],
                                    n_out_range_vec[n]*sizeof(bool),
                                    cudaMemcpyHostToDevice) ) ;
        offset += n_out_range_vec[n] ;
        offset_out += n_out_range_vec[n] ;

        // in-range map for particle n
        CUDA_SAFE_CALL( cudaMemcpy( dev_maps_combined+offset,
                                    dev_maps_updated+offset_updated,
                                    n_in_range_vec[n]*(n_measure+1)*sizeof(Gaussian2D),
                                    cudaMemcpyDeviceToDevice) ) ;
        CUDA_SAFE_CALL( cudaMemcpy( dev_merged_flags_combined+offset,
                                    dev_merged_flags+offset_updated,
                                    n_in_range_vec[n]*(n_measure+1)*sizeof(bool)
                                    ,cudaMemcpyDeviceToDevice ) ) ;
        offset += n_in_range_vec[n]*(n_measure+1) ;
        offset_updated += n_in_range_vec[n]*(n_measure+1) ;
        mapSizes[n] = n_out_range_vec[n] + n_in_range_vec[n]*(n_measure+1) ;
    }

    CUDA_SAFE_CALL( cudaMemcpy( dev_map_sizes, &mapSizes[0],
                                nParticles*sizeof(int),
                                cudaMemcpyHostToDevice ) ) ;
    DEBUG_MSG("launching phdUpdateMergeKernel") ;
    phdUpdateMergeKernel<<<nBlocks,256>>>
        ( dev_maps_combined, dev_maps_merged, dev_n_merged,
          dev_merged_flags_combined, dev_map_sizes, nParticles ) ;
    CUDA_SAFE_THREAD_SYNC() ;

    // allocate outputs
    DEBUG_MSG("Allocating update and merge outputs") ;
    Gaussian2D* maps_merged = (Gaussian2D*)malloc( combined_size ) ;
    Gaussian2D* maps_updated = (Gaussian2D*)malloc( combined_size ) ;
    int* map_sizes_merged = (int*)malloc( nParticles*sizeof(int) ) ;
    char* compatible_z = (char*)malloc( nParticles*n_measure*sizeof(char) ) ;

    // copy outputs
    DEBUG_MSG("cudaMemcpy") ;
    CUDA_SAFE_CALL(
                cudaMemcpy(compatible_z,dev_compatible_z,
                           nParticles*n_measure*sizeof(char),
                           cudaMemcpyDeviceToHost ) ) ;

    CUDA_SAFE_CALL(
                cudaMemcpy( maps_updated, dev_maps_combined,
                            combined_size,
                            cudaMemcpyDeviceToHost ) ) ;

    CUDA_SAFE_CALL(
                cudaMemcpy( maps_merged, dev_maps_merged,
                            combined_size,
                            cudaMemcpyDeviceToHost ) ) ;
    CUDA_SAFE_CALL(
                cudaMemcpy( map_sizes_merged, dev_n_merged,
                            nParticles*sizeof(int),
                            cudaMemcpyDeviceToHost ) ) ;


    ///////////////////////////////////////////////////////////////////////////
    //
    // Save updated maps
    //
    ///////////////////////////////////////////////////////////////////////////

    DEBUG_MSG("Saving update results") ;
    offset_updated = 0 ;
//    int offset_out = 0 ;
    REAL weightSum = 0 ;
    int cn_offset = 0 ;

    for ( int i = 0 ; i < nParticles ; i++ )
    {
//		DEBUG_VAL(map_sizes_merged[i]) ;
//		DEBUG_VAL(offset_updated) ;
        particles.maps[i].assign(maps_merged+offset_updated,
                             maps_merged+offset_updated+map_sizes_merged[i]) ;
//		particles.maps[i].assign( maps_updated+offset_updated,
//								maps_updated+offset_updated+n_in_range_vec[i]*(n_measure+1) ) ;
//		DEBUG_VAL(i) ;
//		DEBUG_VAL(particles.weights[i]) ;
//		DEBUG_VAL(particle_weights[i]) ;
        particles.weights[i] += particle_weights[i]  ;
//		DEBUG_VAL(particles.weights[i]) ;
//		if (particles.weights[i] <= 0 )
//			particles.weights[i] = FLT_MIN ;
        particlesPreMerge.maps[i].assign( maps_updated+offset_updated,
                                          maps_updated+offset_updated+mapSizes[i] ) ;
        offset_updated += mapSizes[i] ;

//		if ( n_out_range > 0 && n_out_range_vec[i] > 0 )
//		{
//			particles.maps[i].insert( particles.maps[i].end(),
//								features_out.begin()+offset_out,
//								features_out.begin()+offset_out+n_out_range_vec[i] ) ;
//			particlesPreMerge.maps[i].insert( particlesPreMerge.maps[i].end(),
//											features_out.begin()+offset_out,
//											features_out.begin()+offset_out+n_out_range_vec[i] ) ;
//			offset_out += n_out_range_vec[i] ;
//		}

        if ( config.filterType == CPHD_TYPE )
        {
            particles.cardinalities[i].assign( cn_update+cn_offset,
                                               cn_update+cn_offset+config.maxCardinality+1) ;
            cn_offset += config.maxCardinality+1 ;
        }
    }

    // save compatible measurement flags
    DEBUG_MSG("Save compatibleZ") ;
    particles.compatibleZ.assign( compatible_z,
                                  compatible_z+nParticles*n_measure) ;

    // normalize particle weights
    weightSum = logSumExp(particles.weights) ;
    DEBUG_VAL(weightSum) ;
    for (int i = 0 ; i < nParticles ; i++ )
    {
        particles.weights[i] -= weightSum ;
//		DEBUG_VAL(particles.weights[i])
    }

    // normalize again
    weightSum = logSumExp(particles.weights) ;
    DEBUG_VAL(weightSum) ;
    for (int i = 0 ; i < nParticles ; i++ )
    {
        particles.weights[i] -= weightSum ;
//		DEBUG_VAL(particles.weights[i])
    }
    weightSum = logSumExp(particles.weights) ;
    DEBUG_VAL(weightSum) ;

    particlesPreMerge.weights = particles.weights ;


//	////////////////////////////////////////////////////////
//	//
//	// Downsample shotgunned particles
//	//
//	////////////////////////////////////////////////////////
//	if ( config.nPredictParticles > 1 )
//	{
//		particles = resampleParticles(particles,config.nParticles) ;
//	}


    // free memory
//	DEBUG_MSG("Freeing in_range") ;
//	free(in_range) ;
    free(maps_updated) ;
//	DEBUG_MSG("freeing maps_merged") ;
    free(maps_merged) ;
//	DEBUG_MSG("Freeing map_sizes_merged") ;
    free(map_sizes_merged) ;
//	DEBUG_MSG("particle_weights") ;
    free(particle_weights) ;
//	DEBUG_MSG("Freeing compatible_z") ;
    free(compatible_z) ;

    CUDA_SAFE_CALL( cudaFree( dev_map_sizes ) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_compatible_z ) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_poses ) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_map_sizes_inrange ) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_maps_merged ) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_maps_updated ) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_n_merged ) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_merged_flags ) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_merged_flags_combined) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_maps_combined ) ) ;
    if ( config.filterType == CPHD_TYPE )
    {
        free(cn_update) ;
    }
    DEBUG_MSG("returning...") ;
    return particlesPreMerge ;
}


ParticleSLAM resampleParticles( ParticleSLAM oldParticles, int n_new_particles )
{
    if ( n_new_particles < 0 )
    {
        n_new_particles = oldParticles.nParticles ;
    }
//	unsigned int nParticles = oldParticles.nParticles ;
    ParticleSLAM newParticles(n_new_particles) ;
    int compatible_z_step = oldParticles.compatibleZ.size()/oldParticles.nParticles ;
    newParticles.compatibleZ.reserve( n_new_particles*compatible_z_step );

    double interval = 1.0/n_new_particles ;
    double r = randu01() * interval ;
    double c = exp(oldParticles.weights[0]) ;
    int i = 0 ;
//	DEBUG_VAL(interval) ;
    for ( int j = 0 ; j < n_new_particles ; j++ )
    {
//		DEBUG_VAL(j) ;
//		DEBUG_VAL(r) ;
//		DEBUG_VAL(c) ;
        while( r > c )
        {
            i++ ;
            // sometimes the weights don't exactly add up to 1, so i can run
            // over the indexing bounds. When this happens, find the most highly
            // weighted particle and fill the rest of the new samples with it
            if ( i >= oldParticles.nParticles || i < 0 || isnan(i) )
            {
                DEBUG_VAL(r) ;
                DEBUG_VAL(c) ;
                double max_weight = -1 ;
                int max_idx = -1 ;
                for ( int k = 0 ; k < oldParticles.nParticles ; k++ )
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
        newParticles.maps[j] = oldParticles.maps[i] ;
        newParticles.compatibleZ.insert(newParticles.compatibleZ.end(),
                                        oldParticles.compatibleZ.begin()+i*compatible_z_step,
                                        oldParticles.compatibleZ.begin()+(i+1)*compatible_z_step ) ;
        newParticles.cardinalities[j] = oldParticles.cardinalities[i] ;
        r += interval ;
    }
    return newParticles ;
}

gaussianMixture computeExpectedMap(ParticleSLAM particles)
// concatenate all particle maps into a single slam particle and then call the
// existing gaussian pruning algorithm ;
{
    DEBUG_MSG("Computing Expected Map") ;
    gaussianMixture concat ;
    int* merged_sizes = (int*)malloc(particles.nParticles*sizeof(int)) ;
    int* map_sizes = (int*)malloc(particles.nParticles*sizeof(int)) ;
    int total_features = 0 ;
    for ( int n = 0 ; n < particles.nParticles ; n++ )
    {
        gaussianMixture map = particles.maps[n] ;
        for ( int i = 0 ; i < map.size() ; i++ )
            map[i].weight *= exp(particles.weights[n]) ;
        concat.insert( concat.end(), map.begin(), map.end() ) ;
        merged_sizes[n] =  map.size() ;
        total_features += map.size() ;
    }

    if ( total_features == 0 )
    {
        DEBUG_MSG("no features") ;
        gaussianMixture expected_map(0) ;
        return expected_map ;
    }
    Gaussian2D* all_features = (Gaussian2D*)malloc( total_features*sizeof(Gaussian2D) ) ;
    std::copy( concat.begin(), concat.end(), all_features ) ;
    bool* merged_flags = (bool*)malloc( total_features*sizeof(sizeof(Gaussian2D) ) ) ;
    std::fill( merged_flags, merged_flags+total_features, false ) ;
    Gaussian2D* maps_out = (Gaussian2D*)malloc( total_features*sizeof(Gaussian2D) ) ;

    Gaussian2D* dev_maps_in = NULL ;
    Gaussian2D* dev_maps_out = NULL ;
    int* dev_merged_sizes = NULL ;
    bool* dev_merged_flags = NULL ;
    int* dev_map_sizes = NULL ;
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_maps_in,
                                total_features*sizeof(Gaussian2D) ) ) ;
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_maps_out,
                                total_features*sizeof(Gaussian2D) ) ) ;
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_merged_sizes,
                                particles.nParticles*sizeof(int) ) ) ;
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_map_sizes,
                                particles.nParticles*sizeof(int) ) ) ;
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_merged_flags,
                                total_features*sizeof(bool) ) ) ;
    for ( int n = particles.nParticles/2 ; n > 0 ; n >>= 1 )
    {
        DEBUG_VAL(n) ;
        for ( int i = 0 ; i < n ; i++ )
            map_sizes[i] = merged_sizes[2*i] + merged_sizes[2*i+1] ;
        CUDA_SAFE_CALL( cudaMemcpy( dev_map_sizes, map_sizes,
                                    n*sizeof(int),
                                    cudaMemcpyHostToDevice ) ) ;
        CUDA_SAFE_CALL( cudaMemcpy( dev_maps_in, all_features,
                                    total_features*sizeof(Gaussian2D),
                                    cudaMemcpyHostToDevice) ) ;
        CUDA_SAFE_CALL( cudaMemcpy( dev_merged_flags, merged_flags,
                                    total_features*sizeof(bool),
                                    cudaMemcpyHostToDevice)) ;
        // kernel launch
        phdUpdateMergeKernel<<<n,256>>>( dev_maps_in, dev_maps_out, dev_merged_sizes,
                                         dev_merged_flags, dev_map_sizes, n ) ;

        CUDA_SAFE_CALL( cudaMemcpy( maps_out, dev_maps_out,
                                    total_features*sizeof(Gaussian2D),
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
    gaussianMixture expected_map(total_features) ;
    std::copy( all_features,all_features+total_features, expected_map.begin() ) ;
    return expected_map ;
}

bool expectedFeaturesPredicate( Gaussian2D g )
{
    return (g.weight <= config.minExpectedFeatureWeight) ;
}

void recoverSlamState(ParticleSLAM particles, ConstantVelocityState& expectedPose,
        gaussianMixture& expectedMap, vector<REAL>& cn_estimate )
{
    if ( particles.nParticles > 1 )
    {
        // calculate the weighted mean of the particle poses
        expectedPose.px = 0 ;
        expectedPose.py = 0 ;
        expectedPose.ptheta = 0 ;
        expectedPose.vx = 0 ;
        expectedPose.vy = 0 ;
        expectedPose.vtheta = 0 ;
        for ( int i = 0 ; i < particles.nParticles ; i++ )
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
        for ( int i = 0 ; i < particles.nParticles ; i++ )
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
            gaussianMixture tmpMap( particles.maps[max_idx] ) ;
            expectedMap = tmpMap ;
        }
        else
        {
            expectedMap = computeExpectedMap( particles ) ;
        }

        cn_estimate = particles.cardinalities[max_idx] ;
    }
    else
    {
        gaussianMixture tmpMap( particles.maps[0] ) ;
        tmpMap.erase(
                remove_if( tmpMap.begin(), tmpMap.end(),
                        expectedFeaturesPredicate),
                tmpMap.end() ) ;
        expectedPose = particles.states[0] ;
        expectedMap = tmpMap ;
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
