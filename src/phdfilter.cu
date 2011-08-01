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
#include "slamparams.h"
#include "cutil.h"
#include <assert.h>
#include <float.h>
//#include "cuPrintf.cu"
#include "matrix.h"
#include "mat.h"

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
phdPredict( ParticleSLAM& particles ) ;

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
ParticleSLAM resampleParticles( ParticleSLAM oldParticles ) ;

extern "C"
void recoverSlamState(ParticleSLAM particles, ConstantVelocityState *expectedPose,
        gaussianMixture *expectedMap) ;

extern "C"
void setDeviceConfig( const SlamConfig& config ) ;
//--- End external function declaration

// SLAM configuration, externally declared
extern SlamConfig config ;

// device memory limit, externally declared
extern size_t deviceMemLimit ;

const char * filename = "data/measurements2.txt" ;
bool breakUpdate = false ;

using namespace std ;

__constant__ RangeBearingMeasurement Z[MAXMEASUREMENTS] ;
__constant__ SlamConfig dev_config ;
//ConstantVelocityModelProps modelProps  = {STDX, STDY,STDTHETA} ;
//ConstantVelocity2DKinematicModel motionModel(modelProps) ;

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
//	REAL sigma[4] ;
	REAL detSigma ;
	REAL sigmaInv[4] = {1,0,0,1} ;
	innov[0] = a.mean[0] - b.mean[0] ;
	innov[1] = a.mean[1] - b.mean[1] ;
//	sigma[0] = a.cov[0] + b.cov[0] ;
//	sigma[1] = a.cov[1] + b.cov[1] ;
//	sigma[2] = a.cov[2] + b.cov[2] ;
//	sigma[3] = a.cov[3] + b.cov[3] ;
//	detSigma = sigma[0]*sigma[3] - sigma[1]*sigma[2] ;
    detSigma = a.cov[0]*a.cov[3] - a.cov[1]*a.cov[2] ;
//	if (detSigma > DBL_EPSILON)
//	{
        sigmaInv[0] = a.cov[3]/detSigma ;
        sigmaInv[1] = -a.cov[1]/detSigma ;
        sigmaInv[2] = -a.cov[2]/detSigma ;
        sigmaInv[3] = a.cov[0]/detSigma ;
//	}
	return  innov[0]*innov[0]*sigmaInv[0] +
			innov[0]*innov[1]*(sigmaInv[1]+sigmaInv[2]) +
			innov[1]*innov[1]*sigmaInv[3] ;
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
__device__ REAL wrapAngle(REAL a)
{
	REAL remainder = fmod(a, 2*M_PI) ;
	if ( remainder > M_PI )
		remainder -= 2*M_PI ;
	else if ( remainder < -M_PI )
		remainder += 2*M_PI ;
	return remainder ;
}

__global__ void
phdPredictKernelVp(AckermanState* particles,
                   AckermanControl control,
                   AckermanNoise* noise)
{
    const int tid = threadIdx.x ;
    const int particleIdx = blockIdx.x*blockDim.x + tid ;
    AckermanState oldState = particles[particleIdx] ;
    AckermanState newState ;
    REAL vc = control.v_encoder/(1-tan(control.alpha)*dev_config.h/dev_config.l) ;
    REAL xc_dot = vc*cos(oldState.ptheta) ;
    REAL yc_dot = vc*sin(oldState.ptheta) ;
    REAL thetac_dot = vc*tan(control.alpha)/dev_config.l ;
    newState.px = oldState.px +
            dev_config.dt*(xc_dot -
            thetac_dot*( dev_config.a*sin(oldState.ptheta) + dev_config.b*cos(oldState.ptheta) )
    ) ;
    newState.py = oldState.py +
            dev_config.dt*(yc_dot -
            thetac_dot*( dev_config.a*cos(oldState.ptheta) - dev_config.b*sin(oldState.ptheta) )
    ) ;
    newState.ptheta = wrapAngle(oldState.ptheta + dev_config.dt*thetac_dot) ;
    particles[particleIdx] = newState ;
}

void
phdPredictVp(ParticleSLAM& particles, AckermanControl control )
{
    // generate random noise values
    int nParticles = particles.nParticles ;
    std::vector<AckermanNoise> noiseVector(nParticles) ;
//    boost::normal_distribution<double> normal_dist ;
//    boost::variate_generator< boost::taus88, boost::normal_distribution<double> > var_gen( rng_g, normal_dist ) ;
    for (unsigned int i = 0 ; i < nParticles ; i++ )
    {
        noiseVector[i].n_alpha = config.std_alpha * randn() ;
        noiseVector[i].n_encoder = config.std_encoder * randn() ;
    }

    // copy to device memory
    cudaEvent_t start, stop ;
    cudaEventCreate( &start ) ;
    cudaEventCreate( &stop ) ;
    cudaEventRecord( start,0 ) ;
    AckermanState* dev_states ;
    AckermanNoise* dev_noise ;

    CUDA_SAFE_CALL(
        cudaMalloc((void**)&dev_states, nParticles*sizeof(AckermanState) )
    ) ;
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&dev_noise, nParticles*sizeof(AckermanNoise) )
    ) ;
    CUDA_SAFE_CALL(
        cudaMemcpy(dev_states, &particles.states[0],
        nParticles*sizeof(ConstantVelocityState),cudaMemcpyHostToDevice)
    ) ;
    CUDA_SAFE_CALL(
        cudaMemcpy(dev_noise, &noiseVector[0],
        nParticles*sizeof(ConstantVelocityNoise), cudaMemcpyHostToDevice)
    ) ;

    // launch the kernel
    int nThreads = min(nParticles,512) ;
    int nBlocks = (nParticles+511)/512 ;
    phdPredictKernelVp
    <<<nBlocks, nThreads>>>
    (dev_states,control,dev_noise) ;

    // copy results from device
    CUDA_SAFE_CALL(
        cudaMemcpy(&particles.states[0], dev_states,
        nParticles*sizeof(ConstantVelocityState),cudaMemcpyDeviceToHost)
    ) ;

    // log time
    cudaEventRecord( stop,0 ) ;
    cudaEventSynchronize( stop ) ;
    float elapsed ;
    cudaEventElapsedTime( &elapsed, start, stop ) ;
    fstream predictTimeFile( "predicttime.log", fstream::out|fstream::app ) ;
    predictTimeFile << elapsed << endl ;
    predictTimeFile.close() ;

    // clean up
    cudaFree( dev_states ) ;
    cudaFree( dev_noise ) ;
}

__global__ void
phdPredictKernel(ConstantVelocityState* particles,
		ConstantVelocityNoise* noise)
{
	const int tid = threadIdx.x ;
	const int particleIdx = blockIdx.x*blockDim.x + tid ;
	ConstantVelocityState oldState = particles[particleIdx] ;
	ConstantVelocityState newState ;
//	typename modelType::stateType newState = mm(particles[particleIdx],*control,noise[particleIdx]) ;
	newState.px = oldState.px +
			DT*(oldState.vx*cos(oldState.ptheta) - oldState.vy*sin(oldState.ptheta))+
			DT2*0.5*(noise[particleIdx].ax*cos(oldState.ptheta) - noise[particleIdx].ay*sin(oldState.ptheta)) ;
	newState.py = oldState.py +
			DT*(oldState.vx*sin(oldState.ptheta) + oldState.vy*cos(oldState.ptheta)) +
			DT2*0.5*(noise[particleIdx].ax*sin(oldState.ptheta) + noise[particleIdx].ay*cos(oldState.ptheta)) ;
	newState.ptheta = wrapAngle(oldState.ptheta + DT*oldState.vtheta + 0.5*DT2*noise[particleIdx].atheta) ;
	newState.vx = oldState.vx + DT*noise[particleIdx].ax ;
	newState.vy = oldState.vy + DT*noise[particleIdx].ay ;
	newState.vtheta = oldState.vtheta + DT*noise[particleIdx].atheta ;
	particles[particleIdx] = newState ;
}

void
phdPredict(ParticleSLAM& particles)
{
	// generate random noise values
    int nParticles = particles.nParticles ;
	std::vector<ConstantVelocityNoise> noiseVector(nParticles) ;
//    boost::normal_distribution<double> normal_dist ;
//    boost::variate_generator< boost::taus88, boost::normal_distribution<double> > var_gen( rng_g, normal_dist ) ;
	for (unsigned int i = 0 ; i < nParticles ; i++ )
	{
        noiseVector[i].ax = 3*STDX * randn() ;
        noiseVector[i].ay = 3*STDY * randn() ;
        noiseVector[i].atheta = 3*STDTHETA * randn() ;
	}

	// copy to device memory
	cudaEvent_t start, stop ;
	cudaEventCreate( &start ) ;
	cudaEventCreate( &stop ) ;
	cudaEventRecord( start,0 ) ;
	ConstantVelocityState* dev_states ;
	ConstantVelocityNoise* dev_noise ;

	cudaMalloc((void**)&dev_states, nParticles*sizeof(ConstantVelocityState) ) ;
	cudaMalloc((void**)&dev_noise, nParticles*sizeof(ConstantVelocityNoise) ) ;
    cudaMemcpy(dev_states, &particles.states[0], nParticles*sizeof(ConstantVelocityState)
		,cudaMemcpyHostToDevice) ;
	cudaMemcpy(dev_noise, &noiseVector[0], nParticles*sizeof(ConstantVelocityNoise)
			, cudaMemcpyHostToDevice) ;

	// launch the kernel
	cudaError errcode ;
	if ( (errcode = cudaPeekAtLastError()) != cudaSuccess )
	{
		cout << "Error before kernel launch: " << cudaGetErrorString(errcode) << endl ;
		exit(0) ;
	}
	int nThreads = min(nParticles,512) ;
	int nBlocks = (nParticles+511)/512 ;
	phdPredictKernel
	<<<nBlocks, nThreads>>>
	(dev_states,dev_noise) ;

	// copy results from device
    cudaMemcpy(&particles.states[0], dev_states,
			nParticles*sizeof(ConstantVelocityState),cudaMemcpyDeviceToHost) ;

	// log time
	cudaEventRecord( stop,0 ) ;
	cudaEventSynchronize( stop ) ;
	float elapsed ;
	cudaEventElapsedTime( &elapsed, start, stop ) ;
	fstream predictTimeFile( "predicttime.log", fstream::out|fstream::app ) ;
	predictTimeFile << elapsed << endl ;
	predictTimeFile.close() ;

	// clean up
	cudaFree( dev_states ) ;
	cudaFree( dev_noise ) ;
}

__global__ void
birthsKernel( ConstantVelocityState* particles, int nParticles, REAL birthWeight,
		RangeBearingMeasurement* ZZ, char* compatibleZ,
		Gaussian2D* births)
{
	__shared__ unsigned int birthCounter ;
	int tid = threadIdx.x ;
	if ( tid == 0 )
		birthCounter = 0 ;
	__syncthreads() ;
	int stateIdx = blockIdx.x  ;
	int offset = stateIdx*blockDim.x ;
	ConstantVelocityState s = particles[stateIdx] ;
	if ( !compatibleZ[offset+tid] )
	{
		unsigned int birthIdx = atomicAdd(&birthCounter, 1) ;
		birthIdx += offset ;
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
		births[birthIdx].cov[0] = J[0]*J[0]*VARRANGE + J[2]*J[2]*VARBEARING ;
		births[birthIdx].cov[1] = J[0]*VARRANGE*J[1] + J[2]*VARBEARING*J[3] ;
		births[birthIdx].cov[2] = births[birthIdx].cov[1] ;
		births[birthIdx].cov[3] = J[1]*J[1]*VARRANGE + J[3]*J[3]*VARBEARING ;
	//	makePositiveDefinite( births[birthIdx].cov) ;
		births[birthIdx].weight = birthWeight ;
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
	ConstantVelocityState* d_particles ;
	RangeBearingMeasurement* d_measurements ;
	char* devCompatibleZ ;
	vector<char> compatibleZ ;
	int nMeasurements = measurements.size() ;
    int nParticles = particles.nParticles ;
    if ( particles.compatibleZ.size() > 0 && config.gatedBirths > 0 )
	{
        compatibleZ = particles.compatibleZ ;
	}
	else
	{
		compatibleZ.assign(nParticles*nMeasurements,0) ;
	}
	size_t particlesSize = nParticles*sizeof(ConstantVelocityState) ;
	size_t measurementsSize = nMeasurements*sizeof(RangeBearingMeasurement) ;
	cudaMalloc( (void**)&d_particles, particlesSize ) ;
	cudaMalloc( (void**)&d_measurements, measurementsSize ) ;
	cudaMalloc( (void**)&devCompatibleZ, nParticles*nMeasurements*sizeof(char) ) ;
    cudaMemcpy( d_particles, &particles.states[0], particlesSize, cudaMemcpyHostToDevice ) ;
	cudaMemcpy( d_measurements, &measurements[0], measurementsSize, cudaMemcpyHostToDevice ) ;
	cudaMemcpy( devCompatibleZ, &compatibleZ[0], nParticles*nMeasurements*sizeof(char), cudaMemcpyHostToDevice ) ;

	// allocate outputs on device
//	DEBUG_MSG("Allocating outputs") ;
	Gaussian2D* d_births ;
	int nBirths = nMeasurements*nParticles ;
	size_t birthsSize = nBirths*sizeof(Gaussian2D) ;
	cudaMalloc( (void**)&d_births, birthsSize ) ;

	// call the kernel
//	DEBUG_MSG("Launching Kernel") ;
	cudaError errcode ;
	if ( (errcode = cudaPeekAtLastError()) != cudaSuccess )
	{
		cout << "Error before kernel launch: " << cudaGetErrorString(errcode) << endl ;
		exit(0) ;
	}
    birthsKernel<<<nParticles, nMeasurements>>>( d_particles, nParticles, config.birthWeight,
			d_measurements, devCompatibleZ, d_births ) ;

	// retrieve outputs from device
//	DEBUG_MSG("Saving birth terms") ;
	Gaussian2D *births = new Gaussian2D[nBirths] ;
	cudaMemcpy( births, d_births, birthsSize, cudaMemcpyDeviceToHost ) ;
	vector<Gaussian2D> birthVector(births, births + nBirths) ;
	vector<Gaussian2D>::iterator i ;
	int n = 0 ;
	for ( i = birthVector.begin() ; i != birthVector.end() ; i+= nMeasurements )
	{
		int offset = n*nMeasurements ;
		int nRealBirths = 0 ;
		for ( int j = 0 ; j<nMeasurements ; j++ )
		{
			if ( !compatibleZ[offset+j] )
				nRealBirths++ ;
		}
        particles.maps[n].insert(particles.maps[n].end(),i,i+nRealBirths) ;
		n++ ;
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
                ConstantVelocityState* poses, bool* inRange, int* nInRange )
{
    int tid = threadIdx.x ;

    // total number of predicted features per block
    int nFeaturesBlock ;
    // number of inrange features in the particle
    __shared__ int nInRangeBlock ;
    // vehicle pose of the thread block
    ConstantVelocityState blockPose ;

    Gaussian2D feature ;
    for ( int p = 0 ; p < nParticles ; p += gridDim.x )
    {
        if ( p < nParticles )
        {
            int predictIdx = 0 ;
            // compute the indexing offset for this particle
            int mapIdx = p + blockIdx.x ;
            for ( int i = 0 ; i < mapIdx ; i++ )
                predictIdx += mapSizes[i] ;
            // particle-wide values
            nInRangeBlock = 0 ;
            blockPose = poses[mapIdx] ;
            nFeaturesBlock = mapSizes[mapIdx] ;
            __syncthreads() ;

            // loop through features
            for ( int i = 0 ; i < nFeaturesBlock ; i += blockDim.x )
            {
                if ( i < nFeaturesBlock )
                {
                    // index of thread feature
                    int featureIdx = predictIdx + tid ;
                    feature = predictedFeatures[featureIdx] ;

                    // default value
                    inRange[featureIdx] = false ;

                    // compute the predicted measurement
                    REAL dx = feature.mean[0] - blockPose.px ;
                    REAL dy = feature.mean[1] - blockPose.py ;
                    REAL r2 = dx*dx + dy*dy ;
                    REAL r = sqrt(r2) ;
                    REAL bearing = wrapAngle(atan2f(dy,dx) - blockPose.ptheta) ;
                    if ( r < MAXRANGE && fabs(bearing) < MAXBEARING )
                    {
                        atomicAdd( &nInRangeBlock, 1 ) ;
                        inRange[featureIdx] = true ;
                    }
                }
            }
            // store nInrange
            if ( tid == 0 )
            {
                nInRange[mapIdx] = nInRangeBlock ;
            }
        }
    }
}

/// perform the gaussian mixture PHD update, and the reduce the resulting mixture
/*!
  PHD update algorithm as in Vo & Ma 2006. Gaussian mixture reduction (merging),
  as in the clustering algorithm by Salmond 1990.
    \param inRangeFeatures Array of in-range Gaussians, with which the PHD
        update will be performed
    \param mapSizes Integer array of sizes of each particle's map
    \param nMeasurements Number of measurements
    \param poses Array of particle poses
    \param compatibleZ char array which will be computed by the kernel.
        Indicates which measurements have been found compatible with an existing
        gaussian.
    \param updatedFeatures Stores the updated Gaussians computed by the kernel
    \param mergedFeatures Stores the post-merge updated Gaussians computed by
        the kernel.
    \param mergedSizes Stores the number of Gaussians left in each map after
        merging. This is required because the same amount of memory is allocated
        for both updatedFeatures and mergedFeatures. The indexing boundaries for
        the maps will be the same, but only the first n gaussians after the
        boundary will be valid for the mergedFeatures array.
    \param mergedFlags Array of booleans used by the merging algorithm to keep
        track of which features have already be merged.
    \param particleWeights New particle weights after PHD update
  */
__global__ void
phdUpdateKernel(Gaussian2D *inRangeFeatures, int* mapSizes,
        int nMeasurements, ConstantVelocityState* poses,
		char* compatibleZ, Gaussian2D *updatedFeatures,
		Gaussian2D *mergedFeatures, int *mergedSizes, bool *mergedFlags,
		REAL *particleWeights )
{
	// shared memory variables
	__shared__ ConstantVelocityState pose ;
	__shared__ REAL sdata[256] ;
	__shared__ Gaussian2D maxFeature ;
	__shared__ REAL wMerge ;
	__shared__ REAL meanMerge[2] ;
	__shared__ REAL covMerge[4] ;
	__shared__ int mergedSize ;
    __shared__ REAL cardinality_predict ;
    __shared__ REAL cardinality_updated ;

	// initialize variables
	int tid = threadIdx.x ;
	int mapIdx ;
	int updateOffset = 0 ;
	int nFeatures = 0 ;
	int nUpdate = 0 ;
	int predictIdx = 0 ;

	// pre-update variables
	REAL featurePd = 0 ;
	REAL dx, dy, r2, r, bearing ;
	REAL J[4] = {0,0,0,0} ;
	REAL K[4] = {0,0,0,0} ;
	REAL sigma[4] = {0,0,0,0} ;
	REAL sigmaInv[4] = {0,0,0,0} ;
	REAL covUpdate[4] = {0,0,0,0} ;
	REAL detSigma = 0 ;
	Gaussian2D feature ;

	// update variables
	RangeBearingMeasurement z ;
	REAL innov[2] = {0,0} ;
	REAL meanUpdate[2] = {0,0} ;
	REAL dist = 0 ;
	REAL logLikelihood = 0 ;
	REAL wPartial = 0 ;
	REAL weightUpdate = 0 ;
	int updateIdx = 0 ;
	// loop over particles
    for ( int p = 0 ; p < dev_config.nParticles ; p += gridDim.x )
	{
		mapIdx = p + blockIdx.x ;
//		mapIdx = blockIdx.x ;
		for ( int i = 0 ; i < mapIdx ; i++ )
			predictIdx += mapSizes[i] ;
		updateOffset = predictIdx*(nMeasurements+1) ;
		predictIdx += tid ;
		nFeatures = mapSizes[mapIdx] ;
		nUpdate = nFeatures*(nMeasurements+1) ;
		if ( tid < nMeasurements )
			compatibleZ[mapIdx*nMeasurements + tid] = 0 ;

        // initialize shared variables
		if ( tid == 0 )
		{
			pose = poses[mapIdx] ;
			particleWeights[mapIdx] = 0 ;
			mergedSize = 0 ;
			mergedSizes[mapIdx] = nUpdate ;
            cardinality_predict = 0.0 ;
            cardinality_updated = 0.0 ;
		}
		__syncthreads() ;

		if ( tid < nFeatures )
		{
			// get the feature corresponding to the current thread
            feature = inRangeFeatures[predictIdx] ;

			/*
			 * PRECOMPUTE UPDATE COMPONENTS
			 */
			// predicted measurement
			dx = feature.mean[0] - pose.px ;
			dy = feature.mean[1] - pose.py ;
			r2 = dx*dx + dy*dy ;
			r = sqrt(r2) ;
			bearing = wrapAngle(atan2f(dy,dx) - pose.ptheta) ;

			// probability of detection
            if ( r <= dev_config.maxRange && fabsf(bearing) <= dev_config.maxBearing )
				featurePd = PD ;
			else
				featurePd = 0 ;

			// measurement jacobian wrt feature
			J[0] = dx/r ;
			J[2] = dy/r ;
			J[1] = -dy/r2 ;
			J[3] = dx/r2 ;

			// BEGIN Maple-Generated expressions
	#define P feature.cov
	#define S sigmaInv
			// innovation covariance
            sigma[0] = (P[0] * J[0] + J[2] * P[1]) * J[0] + (J[0] * P[2] + P[3] * J[2]) * J[2] + pow(dev_config.stdRange,2) ;
			sigma[1] = (P[0] * J[1] + J[3] * P[1]) * J[0] + (J[1] * P[2] + P[3] * J[3]) * J[2];
			sigma[2] = (P[0] * J[0] + J[2] * P[1]) * J[1] + (J[0] * P[2] + P[3] * J[2]) * J[3];
            sigma[3] = (P[0] * J[1] + J[3] * P[1]) * J[1] + (J[1] * P[2] + P[3] * J[3]) * J[3] + pow(dev_config.stdBearing,2) ;

			// enforce symmetry
			makePositiveDefinite(sigma) ;

			detSigma = sigma[0]*sigma[3] - sigma[1]*sigma[2] ;
			sigmaInv[0] = sigma[3]/detSigma ;
			sigmaInv[1] = -sigma[1]/detSigma ;
			sigmaInv[2] = -sigma[2]/detSigma ;
			sigmaInv[3] = sigma[0]/detSigma ;

			// Kalman gain
			K[0] = S[0]*(P[0]*J[0] + P[2]*J[2]) + S[1]*(P[0]*J[1] + P[2]*J[3]) ;
			K[1] = S[0]*(P[1]*J[0] + P[3]*J[2]) + S[1]*(P[1]*J[1] + P[3]*J[3]) ;
			K[2] = S[2]*(P[0]*J[0] + P[2]*J[2]) + S[3]*(P[0]*J[1] + P[2]*J[3]) ;
			K[3] = S[2]*(P[1]*J[0] + P[3]*J[2]) + S[3]*(P[1]*J[1] + P[3]*J[3]) ;

			// Updated covariance (Joseph Form)
			covUpdate[0] = ((1 - K[0] * J[0] - K[2] * J[1]) * P[0] + (-K[0] * J[2] - K[2] * J[3]) * P[1]) * (1 - K[0] * J[0] - K[2] * J[1]) + ((1 - K[0] * J[0] - K[2] * J[1]) * P[2] + (-K[0] * J[2] - K[2] * J[3]) * P[3]) * (-K[0] * J[2] - K[2] * J[3]) + pow(K[0], 2) * VARRANGE + pow(K[2], 2) * VARBEARING;
			covUpdate[2] = ((1 - K[0] * J[0] - K[2] * J[1]) * P[0] + (-K[0] * J[2] - K[2] * J[3]) * P[1]) * (-K[1] * J[0] - K[3] * J[1]) + ((1 - K[0] * J[0] - K[2] * J[1]) * P[2] + (-K[0] * J[2] - K[2] * J[3]) * P[3]) * (1 - K[1] * J[2] - K[3] * J[3]) + K[0] * VARRANGE * K[1] + K[2] * VARBEARING * K[3];
			covUpdate[1] = ((-K[1] * J[0] - K[3] * J[1]) * P[0] + (1 - K[1] * J[2] - K[3] * J[3]) * P[1]) * (1 - K[0] * J[0] - K[2] * J[1]) + ((-K[1] * J[0] - K[3] * J[1]) * P[2] + (1 - K[1] * J[2] - K[3] * J[3]) * P[3]) * (-K[0] * J[2] - K[2] * J[3]) + K[0] * VARRANGE * K[1] + K[2] * VARBEARING * K[3];
			covUpdate[3] = ((-K[1] * J[0] - K[3] * J[1]) * P[0] + (1 - K[1] * J[2] - K[3] * J[3]) * P[1]) * (-K[1] * J[0] - K[3] * J[1]) + ((-K[1] * J[0] - K[3] * J[1]) * P[2] + (1 - K[1] * J[2] - K[3] * J[3]) * P[3]) * (1 - K[1] * J[2] - K[3] * J[3]) + pow(K[1], 2) * VARRANGE + pow(K[3], 2) * VARBEARING;

	#undef P
	#undef S
			/*
			 * END PRECOMPUTE UPDATE COMPONENTS
			 */

			// save the non-detection term
			int nonDetectIdx = updateOffset + tid ;
			REAL nonDetectWeight = feature.weight * (1-featurePd) ;
			updatedFeatures[nonDetectIdx].weight = nonDetectWeight ;
			updatedFeatures[nonDetectIdx].mean[0] = feature.mean[0] ;
			updatedFeatures[nonDetectIdx].mean[1] = feature.mean[1] ;
			updatedFeatures[nonDetectIdx].cov[0] = feature.cov[0] ;
			updatedFeatures[nonDetectIdx].cov[1] = feature.cov[1] ;
			updatedFeatures[nonDetectIdx].cov[2] = feature.cov[2] ;
			updatedFeatures[nonDetectIdx].cov[3] = feature.cov[3] ;
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
            cardinality_predict = sdata[0] ;
		__syncthreads() ;

		/*
		 * LOOP THROUGH MEASUREMENTS AND DO UPDATE
		 */
		for (int i = 0 ; i < nMeasurements ; i++ )
		{
			z = Z[i] ;
			if ( tid < nFeatures)
			{
				// compute innovation
				innov[0] = z.range - r ;
				innov[1] = wrapAngle( z.bearing - bearing ) ;
				// compute updated mean
				meanUpdate[0] = feature.mean[0] + K[0]*innov[0] + K[2]*innov[1] ;
				meanUpdate[1] = feature.mean[1] + K[1]*innov[0] + K[3]*innov[1] ;
				// compute single object likelihood
				dist = innov[0]*innov[0]*sigmaInv[0] +
						innov[0]*innov[1]*(sigmaInv[1] + sigmaInv[2]) +
						innov[1]*innov[1]*sigmaInv[3] ;
				// TODO: does this need to be atomic?
				if ( dist < BIRTH_COMPAT_THRESHOLD )
					compatibleZ[i+mapIdx*nMeasurements] |= 1 ;
				// partially updated weight
				if ( featurePd > DBL_EPSILON )
				{
					logLikelihood = log(featurePd) + log(feature.weight) - 0.5*dist
							- log(2*M_PI) - 0.5*log(detSigma) ;
					wPartial = exp(logLikelihood) ;
				}
				else
				{
					wPartial = 0 ;
				}
			}
			else
			{
				meanUpdate[0] = 0 ;
				meanUpdate[1] = 0 ;
				wPartial = 0 ;
	//			wSum[tid] = 0 ;
			}
			// sum partial weights by parallel reduction
			sumByReduction(sdata, wPartial, tid ) ;

			// finish computing weight
            weightUpdate = wPartial/(sdata[0] + dev_config.clutterDensity ) ;


            if ( tid == 0 && dev_config.particleWeighting == 0 )
			{
                particleWeights[mapIdx] += log(sdata[0] + dev_config.clutterDensity ) ;
			}
			// save updated gaussian
			if ( tid < nFeatures )
			{
				updateIdx = updateOffset + (i+1)*nFeatures + tid ;
				updatedFeatures[updateIdx].weight = weightUpdate ;
				updatedFeatures[updateIdx].mean[0] = meanUpdate[0] ;
				updatedFeatures[updateIdx].mean[1] = meanUpdate[1] ;
				updatedFeatures[updateIdx].cov[0] = covUpdate[0] ;
				updatedFeatures[updateIdx].cov[1] = covUpdate[1] ;
				updatedFeatures[updateIdx].cov[2] = covUpdate[2] ;
				updatedFeatures[updateIdx].cov[3] = covUpdate[3] ;
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

        // Cluster-PHD particle weighting
        if ( dev_config.particleWeighting == 0 )
        {
            if ( tid == 0 )
            {
                particleWeights[mapIdx] -= cardinality_predict ;
                particleWeights[mapIdx] = exp(particleWeights[mapIdx]) ;
            }
        }
        // Vo-EmptyMap particle weighting
        else if ( dev_config.particleWeighting == 1 )
        {
            // updated cardinality = sum of updated weights
            for ( int i = 0 ; i < nUpdate ; i += blockDim.x )
            {
                // to avoid divergence in the reduction function call, load weight
                // into temp variable first
                if ( tid + i < nUpdate )
                    wPartial = updatedFeatures[i+tid].weight ;
                else
                    wPartial = 0.0 ;
                sumByReduction( sdata, wPartial, tid );
                if ( tid == 0 )
                    cardinality_updated += sdata[0] ;
            }

            // compute product of clutter densities by parallel reduction
            if ( tid < nMeasurements )
                wPartial = dev_config.clutterDensity ;
            else
                wPartial = 1.0 ;
            productByReduction( sdata, wPartial, tid ) ;
            if ( tid == 0 )
            {
                particleWeights[mapIdx] = sdata[0] * exp(cardinality_updated - cardinality_predict - dev_config.clutterRate ) ;
            }
        }

		/*
		 * END PHD UPDATE
		 */

		/*
		 * START GAUSSIAN MERGING
		 */
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
			__syncthreads() ;
			// find the maximum feature with parallel reduction
			sdata[tid] = -1 ;
			for ( int i = updateOffset ; i < updateOffset + nUpdate ; i += 256)
			{
				int idx = i + tid ;
				if ( idx < (updateOffset + nUpdate) )
				{
					if( !mergedFlags[idx] )
					{
						if (sdata[tid] == -1 ||
							updatedFeatures[(unsigned int)sdata[tid]].weight < updatedFeatures[idx].weight )
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
						if(updatedFeatures[(unsigned int)sdata[tid]].weight <
						updatedFeatures[(unsigned int)sdata[tid+s]].weight )
						{
							sdata[tid] = sdata[tid+s] ;
						}
					}

				}
				__syncthreads() ;
			}
			if ( sdata[0] == -1 )
				break ;
			else if(tid == 0)
				maxFeature = updatedFeatures[ (unsigned int)sdata[0] ] ;
			__syncthreads() ;

			// find features to merge with max feature
			REAL sval0 = 0 ;
			REAL sval1 = 0 ;
			REAL sval2 = 0 ;
			for ( int i = updateOffset ; i < updateOffset + nUpdate ; i += 256 )
			{
				int idx = tid + i ;
				if ( idx < updateOffset+nUpdate )
				{
					if ( !mergedFlags[idx] )
					{
						feature = updatedFeatures[idx] ;
                        dist = computeMahalDist(maxFeature, feature )
                                *maxFeature.weight*feature.weight
                                /(maxFeature.weight+feature.weight );
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
				wMerge = sdata[0] ;
			__syncthreads() ;
			sumByReduction( sdata, sval1, tid ) ;
			if ( tid == 0 )
				meanMerge[0] = sdata[0]/wMerge ;
			__syncthreads() ;
			sumByReduction( sdata, sval2, tid ) ;
			if ( tid == 0 )
				meanMerge[1] = sdata[0]/wMerge ;
			__syncthreads() ;


			// merge the covariances
			sval0 = 0 ;
			sval1 = 0 ;
			sval2 = 0 ;
	//		sval3 = 0 ;
			for ( int i = updateOffset ; i < updateOffset+nUpdate ; i += 256)
			{
				int idx = tid + i ;
				if ( idx < updateOffset+nUpdate )
				{
					if (!mergedFlags[idx])
					{
						feature = updatedFeatures[idx] ;
                        dist = computeMahalDist(maxFeature, feature )
                                *maxFeature.weight*feature.weight
                                /(maxFeature.weight+feature.weight );
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
				covMerge[0] = sdata[0]/wMerge ;
			__syncthreads() ;
			sumByReduction( sdata, sval1, tid ) ;
			if ( tid == 0 )
			{
				covMerge[1] = sdata[0]/wMerge ;
				covMerge[2] = covMerge[1] ;
			}
			__syncthreads() ;
			sumByReduction( sdata, sval2, tid ) ;
			if ( tid == 0 )
			{
				covMerge[3] = sdata[0]/wMerge ;
				// enforce positive definite matrix
				makePositiveDefinite( covMerge ) ;
				int mergeIdx = updateOffset + mergedSize ;
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
			mergedSizes[mapIdx] = mergedSize ;
	} // end loop over particles
	return ;
}


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
	int nMeasurements = measurements.size() ;
	int *mapSizes = new int[nParticles] ;
	int nThreads = 0 ;
	int totalFeatures = 0 ;
	for ( unsigned int n = 0 ; n < nParticles ; n++ )
	{
        concat.insert( concat.end(), particles.maps[n].begin(),
                particles.maps[n].end()) ;
        mapSizes[n] = particles.maps[n].size() ;
//		DEBUG_VAL(mapSizes[n]) ;

		// keep track of largest map feature count
		if ( mapSizes[n] > nThreads )
			nThreads = mapSizes[n] ;
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
    Gaussian2D* dev_maps ;
    int* dev_map_sizes, *dev_n_in_range ;
    bool* dev_in_range ;
    ConstantVelocityState* dev_poses ;
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
                            totalFeatures*sizeof(bool) ) ) ;
    CUDA_SAFE_CALL(
            cudaMalloc( (void**)&dev_poses,
                        nParticles*sizeof(ConstantVelocityState) ) ) ;

    // copy inputs
    CUDA_SAFE_CALL(
        cudaMemcpy( dev_maps, &concat[0], totalFeatures*sizeof(Gaussian2D),
                    cudaMemcpyHostToDevice )
    ) ;
    CUDA_SAFE_CALL(
        cudaMemcpy( dev_map_sizes, mapSizes, nParticles*sizeof(int),
                    cudaMemcpyHostToDevice )
    ) ;
    CUDA_SAFE_CALL(
                cudaMemcpy(dev_poses,&particles.states[0],
                           nParticles*sizeof(ConstantVelocityState),
                           cudaMemcpyHostToDevice)
    ) ;

    // kernel launch
    computeInRangeKernel<<<nParticles,nThreads>>>
        ( dev_maps, dev_map_sizes, nParticles, dev_poses, dev_in_range, dev_n_in_range) ;

    // allocate outputs
    bool* in_range = (bool*)malloc( totalFeatures*sizeof(bool) ) ;
    vector<int> n_in_range_vec(nParticles) ;

    // copy outputs
    CUDA_SAFE_CALL(
        cudaMemcpy( in_range,dev_in_range,totalFeatures*sizeof(bool),
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
        if (in_range[i])
            features_in[idx_in++] = concat[i] ;
        else
            features_out[idx_out++] = concat[i] ;
    }

    // free memory
    CUDA_SAFE_CALL( cudaFree( dev_in_range ) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_n_in_range ) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_maps ) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_map_sizes ) ) ;
    free(in_range) ;


    ///////////////////////////////////////////////////////////////////////////
    //
    // Do PHD update with only in-range features
    //
    ///////////////////////////////////////////////////////////////////////////

    // check for memory limit for storing measurements in constant mem
	if ( nMeasurements > MAXMEASUREMENTS )
	{
		DEBUG_MSG("Warning: maximum number of measurements per time step exceeded") ;
//		DEBUG_VAL(nMeasurements-MAXMEASUREMENTS) ;
		nMeasurements = MAXMEASUREMENTS ;
	}
    DEBUG_VAL(nMeasurements) ;

    // check device memory limit and split into multiple kernel launches if
    // necessary
    int nUpdate = n_in_range*(nMeasurements+1) ;
    size_t requiredMem =
            n_in_range*sizeof(Gaussian2D) +         // dev_maps
            nParticles*sizeof(int) +                    // dev_map_offsets
            nParticles*sizeof(ConstantVelocityState) +  // dev_poses
            nParticles*nMeasurements*sizeof(char) +     // dev_compatible_z
            nUpdate*sizeof(Gaussian2D) +                // dev_maps_update
            nUpdate*sizeof(Gaussian2D) +                // dev_maps_merged
            nParticles*sizeof(int) +                    // dev_merged_map_sizes
            nParticles*sizeof(REAL) +                   // dev_particle_weights
            nUpdate*sizeof(bool) ;                      // dev_merged_flags
	int numLaunches = ceil( requiredMem/deviceMemLimit ) ;
	if ( numLaunches > 1 )
	{
        DEBUG_MSG( "Device memory limit exceeded, will launch multiple kernels in serial" ) ;
		DEBUG_VAL( numLaunches ) ;
	}

    // perform an (exclusive) prefix scan on the map sizes to determine indexing
    // offsets for each map
    vector<int> map_offsets_in(nParticles,0) ;
    vector<int> map_offsets_out(nParticles,0) ;
    int sum_in = 0 ;
    int sum_out = 0 ;
    for ( int i = 0 ; i < nParticles ; i++ )
    {
        map_offsets_in[i] = sum_in ;
        map_offsets_out[i] = sum_out ;
        sum_in += n_in_range_vec[i] ;
        sum_out += n_out_range_vec[i] ;
    }

    // allocate device memory
    int* dev_n_merged ;
    char* dev_compatible_z ;
    Gaussian2D* dev_maps_updated, *dev_maps_merged ;
    REAL* dev_particle_weights ;
    bool* dev_merged_flags ;
    CUDA_SAFE_CALL(
            cudaMalloc( (void**)&dev_maps,
                        n_in_range*sizeof(Gaussian2D) ) ) ;
    CUDA_SAFE_CALL(
            cudaMalloc( (void**)&dev_map_sizes,
                        nParticles*sizeof(int) ) ) ;
    CUDA_SAFE_CALL(
                cudaMalloc((void**)&dev_compatible_z,
                           nParticles*nMeasurements*sizeof(char) ) ) ;
    CUDA_SAFE_CALL(
                cudaMalloc((void**)&dev_maps_updated,
                           nUpdate*sizeof(Gaussian2D)) ) ;
    CUDA_SAFE_CALL(
                cudaMalloc((void**)&dev_particle_weights,
                           nParticles*sizeof(REAL) ) ) ;
    CUDA_SAFE_CALL(
                cudaMalloc((void**)&dev_maps_merged,
                           nUpdate*sizeof(Gaussian2D)) ) ;
    CUDA_SAFE_CALL(
                cudaMalloc((void**)&dev_n_merged,
                           nParticles*sizeof(int)) ) ;
    CUDA_SAFE_CALL(
                cudaMalloc((void**)&dev_merged_flags,
                           nUpdate*sizeof(bool)) ) ;
    // copy inputs
    CUDA_SAFE_CALL(
        cudaMemcpy( dev_maps, &features_in[0],
                    n_in_range*sizeof(Gaussian2D),
                    cudaMemcpyHostToDevice )
    ) ;
    CUDA_SAFE_CALL(
        cudaMemcpy( dev_map_sizes, &n_in_range_vec[0],
                    nParticles*sizeof(int),
                    cudaMemcpyHostToDevice )
    ) ;
    CUDA_SAFE_CALL(
        cudaMemcpy( dev_map_sizes, &mapSizes[0],
                    nParticles*sizeof(int),
                    cudaMemcpyHostToDevice )
    ) ;

    // launch kernel
    phdUpdateKernel<<<nParticles,256>>>
        ( dev_maps, dev_map_sizes, nMeasurements, dev_poses,
          dev_compatible_z, dev_maps_updated, dev_maps_merged, dev_n_merged,
          dev_merged_flags,dev_particle_weights ) ;

    // allocate outputs
    Gaussian2D* maps_merged = (Gaussian2D*)malloc( nUpdate*sizeof(Gaussian2D) ) ;
    Gaussian2D* maps_updated = (Gaussian2D*)malloc( nUpdate*sizeof(Gaussian2D) ) ;
    int* map_sizes_merged = (int*)malloc( nParticles*sizeof(int) ) ;
    char* compatible_z = (char*)malloc( nParticles*nMeasurements*sizeof(char) ) ;

    // copy outputs
    CUDA_SAFE_CALL(
                cudaMemcpy(compatible_z,dev_compatible_z,
                           nParticles*nMeasurements*sizeof(char),
                           cudaMemcpyDeviceToHost ) ) ;
    CUDA_SAFE_CALL(
                cudaMemcpy(&particles.weights[0],dev_particle_weights,
                           nParticles*sizeof(REAL),cudaMemcpyDeviceToHost ) ) ;

    CUDA_SAFE_CALL(
                cudaMemcpy( maps_updated, dev_maps_updated,
                            nUpdate*sizeof(Gaussian2D),
                            cudaMemcpyDeviceToHost ) ) ;

    CUDA_SAFE_CALL(
                cudaMemcpy( maps_merged, dev_maps_merged,
                            nUpdate*sizeof(Gaussian2D),
                            cudaMemcpyDeviceToHost ) ) ;
    CUDA_SAFE_CALL(
                cudaMemcpy( map_sizes_merged, dev_n_merged,
                            nParticles*sizeof(int),
                            cudaMemcpyDeviceToHost ) ) ;

    // free memory
    CUDA_SAFE_CALL( cudaFree( dev_compatible_z ) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_particle_weights ) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_maps ) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_poses ) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_map_sizes) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_maps_merged ) ) ;
    CUDA_SAFE_CALL( cudaFree( dev_n_merged ) ) ;

    ///////////////////////////////////////////////////////////////////////////
    //
    // Save updated maps
    //
    ///////////////////////////////////////////////////////////////////////////

    int offset_updated = 0 ;
    int offset_out = 0 ;
    REAL weightSum = 0 ;

    for ( int i = 0 ; i < nParticles ; i++ )
    {
        particles.maps[i].assign(maps_merged+offset_updated,
                                 maps_merged+offset_updated+map_sizes_merged[i]) ;
        particles.maps[i].insert( particles.maps[i].end(),
                                  features_out.begin()+offset_out,
                                  features_out.begin()+offset_out+n_out_range_vec[i] ) ;
        weightSum += particles.weights[i] ;

        particlesPreMerge.maps[i].assign( maps_updated+offset_updated,
                                          maps_updated+offset_updated+n_in_range_vec[i]*(nMeasurements+1) ) ;
        particlesPreMerge.maps[i].insert( particlesPreMerge.maps[i].end(),
                                          features_out.begin()+offset_out,
                                          features_out.begin()+offset_out+n_out_range_vec[i] ) ;
        offset_updated += n_in_range_vec[i]*(nMeasurements+1) ;
        offset_out += n_out_range_vec[i] ;
    }

    // save compatible measurement flags
    particles.compatibleZ.assign( compatible_z,
                                  compatible_z+nParticles*nMeasurements) ;

    // normalize particle weights
    for (int i = 0 ; i < nParticles ; i++ )
        particles.weights[i] /= weightSum ;
    particlesPreMerge.weights = particles.weights ;

    // free memory
    free(maps_merged) ;
    free(map_sizes_merged) ;
    CUDA_SAFE_CALL( cudaFree( dev_maps_updated ) ) ;

    return particlesPreMerge ;

//    ///////////////////////////////////////////////////////////////////////////
//    //
//    // Merge Gaussian Mixtures
//    //
//    ///////////////////////////////////////////////////////////////////////////

//    // allocate device memory
//    // maximum number of gaussians ( if no merging happens at all )
//    int n_merged = nUpdate + n_out_range ;
//    Gaussian2D* dev_maps_out_range,dev_maps_merged ;
//    int* dev_offsets_out, dev_n_merged ;
//    CUDA_SAFE_CALL(
//                cudaMalloc((void**)&dev_maps_merged,
//                           n_merged*sizeof(Gaussian2D) ) ) ;
//    CUDA_SAFE_CALL(
//                cudaMalloc((void**)&dev_maps_out_range,
//                           n_out_range*sizeof(Gaussian2D) ) ) ;
//    CUDA_SAFE_CALL(
//                cudaMalloc((void**)&dev_offsets_out,
//                           nParticles*sizeof(int) ) ) ;
//    CUDA_SAFE_CALL(
//                cudaMalloc((void**)&dev_n_merged,
//                           nParticles*sizeof(int) ) ) ;

//    // copy inputs
//    CUDA_SAFE_CALL(
//                cudaMemcpy(dev_maps_out_range,&features_out[0],
//                           n_out_range*sizeof(Gaussian2D),
//                           cudaMemcpyHostToDevice) ) ;
//    CUDA_SAFE_CALL(
//                cudaMemcpy(dev_offsets_out,&map_offsets_out[0],
//                           nParticles*sizeof(int),
//                           cudaMemcpyHostToDevice) ) ;
//    // copy inputs
//    phdUpdateMergeKernel<<<nParticles,256>>>
//        (dev_maps_updated,dev_maps_out_range,dev_offsets_out,dev_maps_merged,
//         dev_n_merged ) ;



//    // free memory
//    CUDA_SAFE_CALL( cudaFree( dev_maps_merged ) ) ;
//    CUDA_SAFE_CALL( cudaFree( dev_maps_out_range ) ) ;
//    CUDA_SAFE_CALL( cudaFree( dev_offsets_out ) ) ;
//    CUDA_SAFE_CALL( cudaFree( dev_n_merged ) ) ;


//// allocate inputs
//	cudaEvent_t start, stop ;
//	cudaEventCreate( &start ) ;
//	cudaEventCreate( &stop ) ;
//	cudaEventRecord( start, 0 ) ;
//	Gaussian2D *hostMaps, *devMaps ;
//	RangeBearingMeasurement *hostZ ;
//	ConstantVelocityState *devPoses ;
//	int *devMapSizes ;
//	hostMaps = &concat[0] ;
//	hostZ = &measurements[0] ;
//	DEBUG_VAL(nMeasurements) ;
//	cudaMalloc((void**)&devMaps, totalFeatures*sizeof(Gaussian2D) ) ;
//	cudaMalloc((void**)&devMapSizes, nParticles*sizeof(int) ) ;
//	cudaMalloc((void**)&devPoses, nParticles*sizeof(ConstantVelocityState) ) ;
//	cudaMemcpy( devMaps, hostMaps, totalFeatures*sizeof(Gaussian2D),
//			cudaMemcpyHostToDevice ) ;
//	cudaMemcpy( devMapSizes, mapSizes, nParticles*sizeof(int),
//			cudaMemcpyHostToDevice ) ;
//    cudaMemcpy( devPoses, &particles.states[0],
//			nParticles*sizeof(ConstantVelocityState), cudaMemcpyHostToDevice ) ;
//	cudaMemcpyToSymbol( Z, hostZ, nMeasurements*sizeof(RangeBearingMeasurement) ) ;

//	// allocate outputs and intermediate variables
//	Gaussian2D *devMapsUpdate, *devMapsMerged ;
//	REAL *devParticleWeights ;
//	int *devMergedMapSizes ;
//	bool *devMergedFlags ;
//	char *devCompatibleZ ;
////	DebugStruct *devDebug ;
//	DEBUG_VAL(nUpdate) ;
//	cudaMalloc( (void**)&devCompatibleZ, nParticles*nMeasurements*sizeof(char) ) ;
//	cudaMalloc( (void**)&devMapsUpdate, nUpdate*sizeof(Gaussian2D) ) ;
//	cudaMalloc( (void**)&devMapsMerged, nUpdate*sizeof(Gaussian2D) ) ;
//	cudaMalloc( (void**)&devMergedMapSizes, nParticles*sizeof(int) ) ;
//	cudaMalloc( (void**)&devParticleWeights, nParticles*sizeof(REAL) ) ;
//	cudaMalloc( (void**)&devMergedFlags, nUpdate*sizeof(bool) ) ;
////	cudaMalloc( (void**)&devDebug, sizeof(DebugStruct) ) ;

//	// launch the kernel
//	cudaError errcode ;
//	if ( (errcode = cudaPeekAtLastError()) != cudaSuccess )
//	{
//		cout << "Error before kernel launch: " << cudaGetErrorString(errcode) << endl ;
//		exit(0) ;
//	}

//	// copy outputs from device
//	int updateOffset = 0 ;
//	int mapSizeUpdate, mapSizeMerged ;
//	REAL *particleWeights = new REAL[nParticles] ;
//	int *mergedMapSizes = new int[nParticles] ;
//	char* compatibleZ = new char[nParticles*nMeasurements] ;
//	Gaussian2D *allUpdated = new Gaussian2D[nUpdate] ;
//	Gaussian2D *allMerged = new Gaussian2D[nUpdate] ;
//	bool *mergedFlags = new bool[nUpdate] ;
//	cudaMemcpy( compatibleZ, devCompatibleZ, nParticles*nMeasurements*sizeof(char),
//				cudaMemcpyDeviceToHost ) ;
//	cudaMemcpy( particleWeights, devParticleWeights, nParticles*sizeof(REAL),
//			cudaMemcpyDeviceToHost ) ;
//	cudaMemcpy( mergedMapSizes, devMergedMapSizes, nParticles*sizeof(int),
//			cudaMemcpyDeviceToHost ) ;
//	cudaMemcpy( allUpdated, devMapsUpdate, nUpdate*sizeof(Gaussian2D),
//				cudaMemcpyDeviceToHost ) ;
//	cudaMemcpy( allMerged, devMapsMerged, nUpdate*sizeof(Gaussian2D),
//			cudaMemcpyDeviceToHost ) ;
//	cudaMemcpy(mergedFlags, devMergedFlags, nUpdate*sizeof(bool),
//			cudaMemcpyDeviceToHost ) ;
//#ifdef DEBUG
//	fstream weightFile("weightUpdates.log",fstream::app|fstream::out) ;
//	for ( int i = 0 ; i < nParticles ; i++ )
//	{
//		weightFile << particleWeights[i] << " " ;
//	}
//	weightFile << endl ;
//	weightFile.close() ;
//#endif


//#ifdef DEBUG
//	for ( int i = 0 ; i < nUpdate ; i++ )
//	{
//		if (allUpdated[i].weight > 3)
//		{
//			printf("Strangely high weight in update: %f (index %d)\n",allUpdated[i].weight, i) ;
//		}
//	}
//#endif

//	// set updated features, and particle weights
//	double weightSum = 0 ;
//	for ( int i = 0 ; i < nParticles ; i++ )
//	{
//		mapSizeUpdate = (nMeasurements+1)*mapSizes[i] ;
//		mapSizeMerged = mergedMapSizes[i] ;
////		DEBUG_VAL(mapSizeUpdate) ;
////		DEBUG_VAL(mapSizeMerged) ;
//		assert( mapSizeMerged <= mapSizeUpdate ) ;
//		for ( int j = 0 ; j < mapSizeMerged ; j++ )
//		{
//			if( allMerged[updateOffset+j].weight > 3)
//			{
//				printf("Strangely high weight from merging: %f (index %d)\n",allMerged[updateOffset+j].weight,j+updateOffset) ;
//			}
//		}
//        particles.maps[i].assign(allMerged+updateOffset,allMerged+updateOffset+mapSizeMerged ) ;
//        particles.weights[i] *= particleWeights[i] ;
//		particlesPreMerge.maps[i].assign( allUpdated+updateOffset, allUpdated+updateOffset+mapSizeUpdate ) ;
//        weightSum += particles.weights[i] ;
//		updateOffset += mapSizeUpdate ;
//	}
////	DEBUG_VAL(weightSum) ;
//	// normalize particle weights
//	for (int i = 0 ; i < nParticles ; i++ )
//	{
//        particles.weights[i] /= weightSum ;
////		DEBUG_VAL(particles->weights[i]) ;
//	}

//	// save the measurement compatibility for next birth step
//    particles.compatibleZ.assign( compatibleZ, compatibleZ+nParticles*nMeasurements) ;

//	cudaEventRecord( stop, 0 ) ;
//	cudaEventSynchronize( stop ) ;
//	float elapsedTime ;
//	cudaEventElapsedTime( &elapsedTime, start, stop ) ;
//	fstream updateTimeFile("updatetime.log", fstream::out|fstream::app ) ;
//	updateTimeFile << elapsedTime << endl ;
//	updateTimeFile.close() ;

//	delete[] particleWeights ;
//	delete[] mapSizes ;
//	delete[] allUpdated ;
//	delete[] allMerged ;
//	delete[] mergedMapSizes ;
//	delete[] mergedFlags ;
//	delete[] compatibleZ ;
//	cudaFree( devMergedFlags ) ;
//	cudaFree( devMergedMapSizes ) ;
//	cudaFree( devMaps ) ;
//	cudaFree( devMapSizes ) ;
//	cudaFree( devMapsUpdate ) ;
//	cudaFree( devMapsMerged ) ;
//	cudaFree( devParticleWeights ) ;
//	cudaFree( devPoses ) ;
//	cudaFree( devCompatibleZ ) ;
//	return particlesPreMerge ;
}

__global__ void
mergeKernelSingle( Gaussian2D *features, int nFeatures,
		Gaussian2D *mergedFeatures, bool *mergedFlags, int *nMerged)
{
    __shared__ REAL wSum[256] ;
    __shared__ REAL meanSum[256][2] ;
    __shared__ REAL covSum[256][4] ;
	__shared__ REAL wMerge ;
	__shared__ REAL meanMerge[2] ;
	__shared__ REAL covMerge[4] ;
	__shared__ Gaussian2D maxFeature ;
	int tid = threadIdx.x  ;

	// initialize variables
	for ( int i = 0 ; i < nFeatures ; i += blockDim.x )
	{
		int idx = tid + i ;
		if ( idx < nFeatures )
			mergedFlags[idx] = false ;
	}
	if ( tid == 0 )
		*nMerged = 0 ;

	Gaussian2D feature ;
	REAL innov[2] ;
	REAL dist ;
	while(true)
	{
		// find the maximum feature with parallel reduction
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
		wSum[tid] = -1 ;
		meanSum[tid][0] = -1 ;
		meanSum[tid][1] = -1 ;
		covSum[tid][0] = -1 ;
		covSum[tid][1] = -1 ;
		covSum[tid][2] = -1 ;
		covSum[tid][3] = -1 ;
		for ( int i = 0 ; i < nFeatures ; i+=blockDim.x )
		{
			int idx = tid + i ;
			if ( idx < nFeatures )
			{
				feature = features[idx] ;
				if( !mergedFlags[idx] && wSum[tid] < feature.weight )
				{
					wSum[tid] = feature.weight ;
					meanSum[tid][0] = feature.mean[0] ;
					meanSum[tid][1] = feature.mean[1] ;
					covSum[tid][0] = feature.cov[0] ;
					covSum[tid][1] = feature.cov[1] ;
					covSum[tid][2] = feature.cov[2] ;
					covSum[tid][3] = feature.cov[3] ;
				}
			}
		}
		__syncthreads() ;
//		if (tid < 256 && wSum[tid] < wSum[tid+256] )
//		{
//			wSum[tid] = wSum[tid+256] ;
//			meanSum[tid][0] = meanSum[tid+256][0] ;
//			meanSum[tid][1] = meanSum[tid+256][1] ;
//			covSum[tid][0] = covSum[tid+256][0] ;
//			covSum[tid][1] = covSum[tid+256][1] ;
//			covSum[tid][2] = covSum[tid+256][2] ;
//			covSum[tid][3] = covSum[tid+256][3] ;
//		}
		__syncthreads() ;
		if (tid < 128 && wSum[tid] < wSum[tid+128] )
		{
			wSum[tid] = wSum[tid+128] ;
			meanSum[tid][0] = meanSum[tid+128][0] ;
			meanSum[tid][1] = meanSum[tid+128][1] ;
			covSum[tid][0] = covSum[tid+128][0] ;
			covSum[tid][1] = covSum[tid+128][1] ;
			covSum[tid][2] = covSum[tid+128][2] ;
			covSum[tid][3] = covSum[tid+128][3] ;
		}
		__syncthreads() ;
		if (tid < 64 && wSum[tid] < wSum[tid+64] )
		{
			wSum[tid] = wSum[tid+64] ;
			meanSum[tid][0] = meanSum[tid+64][0] ;
			meanSum[tid][1] = meanSum[tid+64][1] ;
			covSum[tid][0] = covSum[tid+64][0] ;
			covSum[tid][1] = covSum[tid+64][1] ;
			covSum[tid][2] = covSum[tid+64][2] ;
			covSum[tid][3] = covSum[tid+64][3] ;
		}
		__syncthreads() ;

		if (tid < 32 && wSum[tid] < wSum[tid+32] )
		{
			wSum[tid] = wSum[tid+32] ;
			meanSum[tid][0] = meanSum[tid+32][0] ;
			meanSum[tid][1] = meanSum[tid+32][1] ;
			covSum[tid][0] = covSum[tid+32][0] ;
			covSum[tid][1] = covSum[tid+32][1] ;
			covSum[tid][2] = covSum[tid+32][2] ;
			covSum[tid][3] = covSum[tid+32][3] ;
		}
		__syncthreads() ;

		if (tid < 16 && wSum[tid] < wSum[tid+16] )
		{
			wSum[tid] = wSum[tid+16] ;
			meanSum[tid][0] = meanSum[tid+16][0] ;
			meanSum[tid][1] = meanSum[tid+16][1] ;
			covSum[tid][0] = covSum[tid+16][0] ;
			covSum[tid][1] = covSum[tid+16][1] ;
			covSum[tid][2] = covSum[tid+16][2] ;
			covSum[tid][3] = covSum[tid+16][3] ;
		}
		__syncthreads() ;

		if (tid < 8 && wSum[tid] < wSum[tid+8] )
		{
			wSum[tid] = wSum[tid+8] ;
			meanSum[tid][0] = meanSum[tid+8][0] ;
			meanSum[tid][1] = meanSum[tid+8][1] ;
			covSum[tid][0] = covSum[tid+8][0] ;
			covSum[tid][1] = covSum[tid+8][1] ;
			covSum[tid][2] = covSum[tid+8][2] ;
			covSum[tid][3] = covSum[tid+8][3] ;
		}
		__syncthreads() ;

		if ( tid < 4 && wSum[tid] < wSum[tid+4] )
		{
			wSum[tid] = wSum[tid+4] ;
			meanSum[tid][0] = meanSum[tid+4][0] ;
			meanSum[tid][1] = meanSum[tid+4][1] ;
			covSum[tid][0] = covSum[tid+4][0] ;
			covSum[tid][1] = covSum[tid+4][1] ;
			covSum[tid][2] = covSum[tid+4][2] ;
			covSum[tid][3] = covSum[tid+4][3] ;
		}
		__syncthreads() ;

		if ( tid < 2 && wSum[tid] < wSum[tid+2] )
		{
			wSum[tid] = wSum[tid+2] ;
			meanSum[tid][0] = meanSum[tid+2][0] ;
			meanSum[tid][1] = meanSum[tid+2][1] ;
			covSum[tid][0] = covSum[tid+2][0] ;
			covSum[tid][1] = covSum[tid+2][1] ;
			covSum[tid][2] = covSum[tid+2][2] ;
			covSum[tid][3] = covSum[tid+2][3] ;
		}
		__syncthreads() ;

		if ( tid < 1 && wSum[tid] < wSum[tid+1] )
		{
			wSum[tid] = wSum[tid+1] ;
			meanSum[tid][0] = meanSum[tid+1][0] ;
			meanSum[tid][1] = meanSum[tid+1][1] ;
			covSum[tid][0] = covSum[tid+1][0] ;
			covSum[tid][1] = covSum[tid+1][1] ;
			covSum[tid][2] = covSum[tid+1][2] ;
			covSum[tid][3] = covSum[tid+1][3] ;
		}
		__syncthreads() ;
		if ( tid == 0 )
		{
			maxFeature.weight = wSum[0] ;
			maxFeature.mean[0] = meanSum[0][0] ;
			maxFeature.mean[1] = meanSum[0][1] ;
			maxFeature.cov[0] = covSum[0][0] ;
			maxFeature.cov[1] = covSum[0][1] ;
			maxFeature.cov[2] = covSum[0][2] ;
			maxFeature.cov[3] = covSum[0][3] ;
		}
		__syncthreads() ;

		// exit when no more valid features are found
		if ( maxFeature.weight == -1 )
			break ;

		// find features to merge with max feature
		wSum[tid] = 0 ;
		meanSum[tid][0] = 0 ;
		meanSum[tid][1] = 0 ;
		for ( int i = 0 ; i < nFeatures ; i+=blockDim.x )
		{
			int idx = tid + i ;
			if ( idx < nFeatures )
			{
				if ( !mergedFlags[idx] )
				{
					feature = features[idx] ;
					dist = computeMahalDist(maxFeature, feature) ;
                    if ( dist < dev_config.minSeparation )
					{
						wSum[tid] += feature.weight ;
						meanSum[tid][0] += feature.mean[0]*feature.weight ;
						meanSum[tid][1] += feature.mean[1]*feature.weight ;
					}
				}
			}
		}
		__syncthreads() ;
		/*
		 * sum the weights and means by parallel reduction
		 */
//		if ( tid < 256 )
//		{
//			wSum[tid] += wSum[tid+256] ;
//			meanSum[tid][0] += meanSum[tid+256][0] ;
//			meanSum[tid][1] += meanSum[tid+256][1] ;
//		}
		__syncthreads() ;
		if ( tid < 128 )
		{
			wSum[tid] += wSum[tid+128] ;
			meanSum[tid][0] += meanSum[tid+128][0] ;
			meanSum[tid][1] += meanSum[tid+128][1] ;
		}
		__syncthreads() ;
		if ( tid < 64 )
		{
			wSum[tid] += wSum[tid+64] ;
			meanSum[tid][0] += meanSum[tid+64][0] ;
			meanSum[tid][1] += meanSum[tid+64][1] ;
		}
		__syncthreads() ;
		// last warp...
		if ( tid < 32 )
		{
			wSum[tid] += wSum[tid+32] ;
			meanSum[tid][0] += meanSum[tid+32][0] ;
			meanSum[tid][1] += meanSum[tid+32][1] ;

			wSum[tid] += wSum[tid+16] ;
			meanSum[tid][0] += meanSum[tid+16][0] ;
			meanSum[tid][1] += meanSum[tid+16][1] ;

			wSum[tid] += wSum[tid+8] ;
			meanSum[tid][0] += meanSum[tid+8][0] ;
			meanSum[tid][1] += meanSum[tid+8][1] ;

			wSum[tid] += wSum[tid+4] ;
			meanSum[tid][0] += meanSum[tid+4][0] ;
			meanSum[tid][1] += meanSum[tid+4][1] ;

			wSum[tid] += wSum[tid+2] ;
			meanSum[tid][0] += meanSum[tid+2][0] ;
			meanSum[tid][1] += meanSum[tid+2][1] ;

			wSum[tid] += wSum[tid+1] ;
			meanSum[tid][0] += meanSum[tid+1][0] ;
			meanSum[tid][1] += meanSum[tid+1][1] ;
		}
		__syncthreads() ;
		// store the merged weight and mean
		if ( tid == 0 )
		{
			wMerge = wSum[0] ;
			meanMerge[0] = meanSum[0][0]/wMerge ;
			meanMerge[1] = meanSum[0][1]/wMerge ;
		}
		__syncthreads() ;

		// merge the covariances
		covSum[tid][0] = 0 ;
		covSum[tid][1] = 0 ;
		covSum[tid][2] = 0 ;
		covSum[tid][3] = 0 ;
		for ( int i = 0 ; i < nFeatures ; i+=blockDim.x )
		{
			int idx = tid + i ;
			if ( idx < nFeatures )
			{
				if (!mergedFlags[idx])
				{
					feature = features[idx] ;
					dist = computeMahalDist(maxFeature, feature) ;
                    if ( dist < dev_config.minSeparation )
					{
						innov[0] = meanMerge[0] - feature.mean[0] ;
						innov[1] = meanMerge[1] - feature.mean[1] ;
						covSum[tid][0] += (feature.cov[0] + innov[0]*innov[0])*feature.weight ;
						covSum[tid][1] += (feature.cov[1] + innov[0]*innov[1])*feature.weight ;
						covSum[tid][2] += (feature.cov[2] + innov[0]*innov[1])*feature.weight ;
						covSum[tid][3] += (feature.cov[3] + innov[1]*innov[1])*feature.weight ;
						mergedFlags[idx] = true ;
					}
				}
			}
		}
		__syncthreads() ;

//		if ( tid < 256 )
//		{
//			covSum[tid][0] += covSum[tid+256][0] ;
//			covSum[tid][1] += covSum[tid+256][1] ;
//			covSum[tid][2] += covSum[tid+256][2] ;
//			covSum[tid][3] += covSum[tid+256][3] ;
//		}
		__syncthreads() ;

		if ( tid < 128 )
		{
			covSum[tid][0] += covSum[tid+128][0] ;
			covSum[tid][1] += covSum[tid+128][1] ;
			covSum[tid][2] += covSum[tid+128][2] ;
			covSum[tid][3] += covSum[tid+128][3] ;
		}
		__syncthreads() ;

		if ( tid < 64 )
		{
			covSum[tid][0] += covSum[tid+64][0] ;
			covSum[tid][1] += covSum[tid+64][1] ;
			covSum[tid][2] += covSum[tid+64][2] ;
			covSum[tid][3] += covSum[tid+64][3] ;
		}
		__syncthreads() ;

		// last warp...
		if ( tid < 32 )
		{
			covSum[tid][0] += covSum[tid+32][0] ;
			covSum[tid][1] += covSum[tid+32][1] ;
			covSum[tid][2] += covSum[tid+32][2] ;
			covSum[tid][3] += covSum[tid+32][3] ;

			covSum[tid][0] += covSum[tid+16][0] ;
			covSum[tid][1] += covSum[tid+16][1] ;
			covSum[tid][2] += covSum[tid+16][2] ;
			covSum[tid][3] += covSum[tid+16][3] ;

			covSum[tid][0] += covSum[tid+8][0] ;
			covSum[tid][1] += covSum[tid+8][1] ;
			covSum[tid][2] += covSum[tid+8][2] ;
			covSum[tid][3] += covSum[tid+8][3] ;

			covSum[tid][0] += covSum[tid+4][0] ;
			covSum[tid][1] += covSum[tid+4][1] ;
			covSum[tid][2] += covSum[tid+4][2] ;
			covSum[tid][3] += covSum[tid+4][3] ;

			covSum[tid][0] += covSum[tid+2][0] ;
			covSum[tid][1] += covSum[tid+2][1] ;
			covSum[tid][2] += covSum[tid+2][2] ;
			covSum[tid][3] += covSum[tid+2][3] ;

			covSum[tid][0] += covSum[tid+1][0] ;
			covSum[tid][1] += covSum[tid+1][1] ;
			covSum[tid][2] += covSum[tid+1][2] ;
			covSum[tid][3] += covSum[tid+1][3] ;
		}
		__syncthreads() ;
		// store the merged covariance
		if ( tid == 0 )
		{
			covMerge[0] = covSum[0][0]/wMerge ;
			covMerge[1] = covSum[0][1]/wMerge ;
			covMerge[2] = covSum[0][2]/wMerge ;
			covMerge[3] = covSum[0][3]/wMerge ;
		}

		// store merged gaussian in output array
		if ( tid == 0 )
		{
			int mergeIdx = *nMerged ;
			mergedFeatures[mergeIdx].weight = wMerge ;
			mergedFeatures[mergeIdx].mean[0] = meanMerge[0] ;
			mergedFeatures[mergeIdx].mean[1] = meanMerge[1] ;
			mergedFeatures[mergeIdx].cov[0] = covMerge[0] ;
			mergedFeatures[mergeIdx].cov[1] = covMerge[1] ;
			mergedFeatures[mergeIdx].cov[2] = covMerge[2] ;
			mergedFeatures[mergeIdx].cov[3] = covMerge[3] ;
			*nMerged += 1 ;
		}
		__syncthreads() ;
	}
}



void mergeGaussianMixture(gaussianMixture *GM)
{
	int nFeatures = GM->size() ;
	int nBlocks = (nFeatures + 255)/256 ;
	DEBUG_VAL(nFeatures) ;
	DEBUG_VAL(nBlocks) ;
	size_t GMSize = nFeatures*sizeof(Gaussian2D) ;
	Gaussian2D *devGaussians, *devGaussiansMerged ;

	cudaEvent_t start, stop ;
	cudaEventCreate( &start ) ;
	cudaEventCreate( &stop ) ;
	cudaEventRecord( start, 0 ) ;

//	DEBUG_MSG("Allocating inputs") ;
    CUDA_SAFE_CALL( cudaMalloc( (void**)&devGaussians, GMSize ) ) ;
    CUDA_SAFE_CALL(
                cudaMemcpy( devGaussians, &GM->front(), GMSize,
                            cudaMemcpyHostToDevice ) ) ;

//	DEBUG_MSG("Allocating outputs") ;
	int *devSizeMerged ;
	bool *devMergedFlags ;
    CUDA_SAFE_CALL( cudaMalloc( (void**)&devGaussiansMerged, GMSize ) ) ;
    CUDA_SAFE_CALL( cudaMalloc( (void**)&devSizeMerged, sizeof(int) ) ) ;
    CUDA_SAFE_CALL( cudaMalloc( (void**)&devMergedFlags, nFeatures*sizeof(bool) ) ) ;

	DEBUG_MSG("Launching kernel...") ;
	// kernel call
    mergeKernelSingle<<<1,256>>>( devGaussians, nFeatures,
			devGaussiansMerged, devMergedFlags, devSizeMerged ) ;;

	DEBUG_MSG("Copying outputs from device") ;
	int nMerged  ;
	Gaussian2D *maxFeatures = new Gaussian2D[nBlocks] ;
	bool *hostMergedFlags = new bool[nFeatures] ;
//	cudaMemcpy( maxFeatures, devMaxFeature, nBlocks*sizeof(Gaussian2D), cudaMemcpyDeviceToHost ) ;
    CUDA_SAFE_CALL(
                cudaMemcpy( &nMerged, devSizeMerged, sizeof(int),
                            cudaMemcpyDeviceToHost ) ) ;
    CUDA_SAFE_CALL(
                cudaMemcpy( hostMergedFlags, devMergedFlags,
                            nFeatures*sizeof(bool), cudaMemcpyDeviceToHost ) ) ;
	DEBUG_VAL(nMerged) ;

	gaussianMixture gaussiansMerged(nMerged) ;
	Gaussian2D *tmp = new Gaussian2D[nMerged] ;
    CUDA_SAFE_CALL(
                cudaMemcpy( tmp, devGaussiansMerged, nMerged*sizeof(Gaussian2D),
                            cudaMemcpyDeviceToHost ) ) ;
	GM->assign(tmp, tmp+nMerged) ;

	cudaEventRecord( stop, 0 ) ;
	cudaEventSynchronize( stop ) ;
	float elapsed ;
	cudaEventElapsedTime( &elapsed, start, stop ) ;
	fstream mergeTimeFile("mergetime.log", fstream::out|fstream::app ) ;
	mergeTimeFile << elapsed << endl ;
	mergeTimeFile.close() ;

	delete[] tmp ;
	delete[] hostMergedFlags ;
	cudaFree(devSizeMerged) ;
	cudaFree(devGaussians) ;
	cudaFree(devGaussiansMerged) ;
	cudaFree( devMergedFlags ) ;
}


ParticleSLAM resampleParticles( ParticleSLAM oldParticles )
{
//    boost::uniform_01<> uni_dist ;
	unsigned int nParticles = oldParticles.nParticles ;
	ParticleSLAM newParticles(nParticles) ;
	REAL interval = 1.0/nParticles ;
    REAL r = randu01() * interval ;
	REAL c = oldParticles.weights[0] ;
	int i = 0 ;
//	DEBUG_VAL(interval) ;
	for ( int j = 0 ; j < nParticles ; j++ )
	{
//		DEBUG_VAL(j) ;
//		DEBUG_VAL(r) ;
//		DEBUG_VAL(c) ;
		while( r > c )
		{
			i++ ;
			c += oldParticles.weights[i] ;
//			DEBUG_VAL(c) ;
		}
		newParticles.weights[j] = interval ;
		newParticles.states[j] = oldParticles.states[i] ;
		newParticles.maps[j].assign(oldParticles.maps[j].begin(),
				oldParticles.maps[j].end()) ;
		r += interval ;
	}
	return newParticles ;
}

gaussianMixture computeExpectedMap(ParticleSLAM particles)
// concatenate all particle maps into a single slam particle and then call the
// existing gaussian pruning algorithm ;
{
	gaussianMixture concat ;
	int totalFeatures = 0 ;
	DEBUG_MSG("Concatenating Maps") ;
	for ( int n = 0 ; n < particles.nParticles ; n++ )
	{
		int oldEnd = concat.size() ;
		concat.insert(concat.end(),
				particles.maps[n].begin(),particles.maps[n].end() ) ;
		int nFeatures = particles.maps[n].size() ;
		for ( int i = oldEnd ; i < oldEnd + nFeatures; i++ )
		{
			concat[i].weight *= particles.weights[n] ;
		}
		totalFeatures += particles.maps[n].size() ;
	}
	if ( totalFeatures == 0 )
	{
		DEBUG_MSG("All maps are empty") ;
	}
	else
	{
		DEBUG_MSG("Merging concatenated maps") ;
		mergeGaussianMixture(&concat) ;
	}
	return concat ;
}

bool expectedFeaturesPredicate( Gaussian2D g )
{
    return (g.weight <= config.minExpectedFeatureWeight) ;
}

void recoverSlamState(ParticleSLAM particles, ConstantVelocityState *expectedPose,
		gaussianMixture *expectedMap)
{
	if ( particles.nParticles > 1 )
	{
		// calculate the weighted mean of the particle poses
		expectedPose->px = 0 ;
		expectedPose->py = 0 ;
		expectedPose->ptheta = 0 ;
		expectedPose->vx = 0 ;
		expectedPose->vy = 0 ;
		expectedPose->vtheta = 0 ;
		for ( int i = 0 ; i < particles.nParticles ; i++ )
		{
			expectedPose->px += particles.weights[i]*particles.states[i].px ;
			expectedPose->py += particles.weights[i]*particles.states[i].py ;
			expectedPose->ptheta += particles.weights[i]*particles.states[i].ptheta ;
			expectedPose->vx += particles.weights[i]*particles.states[i].vx ;
			expectedPose->vy += particles.weights[i]*particles.states[i].vy ;
			expectedPose->vtheta += particles.weights[i]*particles.states[i].vtheta ;
		}
		gaussianMixture tmpMap = computeExpectedMap(particles) ;
		tmpMap.erase(
						remove_if( tmpMap.begin(), tmpMap.end(),
								expectedFeaturesPredicate),
						tmpMap.end() ) ;
		*expectedMap = tmpMap ;
	}
	else
	{
		gaussianMixture tmpMap( particles.maps[0] ) ;
		tmpMap.erase(
				remove_if( tmpMap.begin(), tmpMap.end(),
						expectedFeaturesPredicate),
				tmpMap.end() ) ;
		*expectedPose = particles.states[0] ;
		*expectedMap = tmpMap ;
	}
}

/// copy the configuration structure to constant device memory
void
setDeviceConfig( const SlamConfig& config )
{
    cudaMemcpyToSymbol( "dev_config", &config, sizeof(SlamConfig) ) ;
}

//void
//writeParticlesMat(ParticleSLAM particles, int t = -1, const char* filename="particles")
//{
//	// create the filename
//	std::string particlesFilename(filename) ;
//	if ( t >= 0 )
//	{
//		char timeStep[8] ;
//		sprintf(timeStep,"%d",t) ;
//		particlesFilename += timeStep ;
//	}
//	particlesFilename += ".mat" ;

//	// load particles into mxArray object
//	mwSize nParticles = particles.nParticles ;

//	mxArray* states = mxCreateNumericMatrix(6,nParticles,mxDOUBLE_CLASS,mxREAL) ;
//	double* statesArray = (double*)mxCalloc(nParticles*6,sizeof(double));
//	int i = 0 ;
//	for ( int p = 0 ; p < nParticles ; p++ )
//	{
//		statesArray[i+0] = particles.states[p].px ;
//		statesArray[i+1] = particles.states[p].py ;
//		statesArray[i+2] = particles.states[p].ptheta ;
//		statesArray[i+3] = particles.states[p].vx ;
//		statesArray[i+4] = particles.states[p].vy ;
//		statesArray[i+5] = particles.states[p].vtheta ;
//		i+=6 ;
//	}
//	mxFree(mxGetPr(states)) ;
//	mxSetPr(states,statesArray) ;

//	mxArray* weights = mxCreateNumericMatrix(nParticles,1,mxDOUBLE_CLASS,mxREAL) ;
//	double* weightsArray = (double*)mxCalloc(nParticles,sizeof(double)) ;
//    std::copy(particles.weights.begin(),particles,weights.end(),weightsArray ) ;
//	mxFree(mxGetPr(weights)) ;
//	mxSetPr(weights,weightsArray) ;

//	const char* mapFieldNames[] = {"weights","means","covs"} ;
//	mxArray* maps = mxCreateStructMatrix(nParticles,1,3,mapFieldNames) ;
//	mwSize covDims[3] = {2,2,2} ;
//	mxArray* mapWeights ;
//	mxArray* mapMeans ;
//	mxArray* mapCovs ;
//	for ( int p = 0 ; p < nParticles ; p++ )
//	{
//		gaussianMixture map = particles.maps[p] ;
//		mwSize mapSize = map.size() ;
//		covDims[2] = mapSize ;
//		mapWeights = mxCreateNumericMatrix(1,mapSize,mxDOUBLE_CLASS,mxREAL) ;
//		mapMeans = mxCreateNumericMatrix(2,mapSize,mxDOUBLE_CLASS,mxREAL) ;
//		mapCovs = mxCreateNumericArray(3,covDims,mxDOUBLE_CLASS,mxREAL) ;
//		if ( mapSize > 0 )
//		{
//			for ( int j = 0 ; j < mapSize ; j++ )
//			{
//				mxGetPr( mapWeights )[j] = map[j].weight ;
//				mxGetPr( mapMeans )[2*j+0] = map[j].mean[0] ;
//				mxGetPr( mapMeans )[2*j+1] = map[j].mean[1] ;
//				mxGetPr( mapCovs )[4*j+0] = map[j].cov[0] ;
//				mxGetPr( mapCovs )[4*j+1] = map[j].cov[1] ;
//				mxGetPr( mapCovs )[4*j+2] = map[j].cov[2] ;
//				mxGetPr( mapCovs )[4*j+3] = map[j].cov[3] ;
//			}
//		}
//		mxSetFieldByNumber( maps, p, 0, mapWeights ) ;
//		mxSetFieldByNumber( maps, p, 1, mapMeans ) ;
//		mxSetFieldByNumber( maps, p, 2, mapCovs ) ;
//	}

//	const char* particleFieldNames[] = {"states","weights","maps"} ;
//	mxArray* mxParticles = mxCreateStructMatrix(1,1,3,particleFieldNames) ;
//	mxSetFieldByNumber( mxParticles, 0, 0, states ) ;
//	mxSetFieldByNumber( mxParticles, 0, 1, weights ) ;
//	mxSetFieldByNumber( mxParticles, 0, 2, maps ) ;

//	// write to mat file
//	MATFile* matfile = matOpen( particlesFilename.c_str(), "w" ) ;
//	matPutVariable( matfile, "particles", mxParticles ) ;
//	matClose(matfile) ;

//	// clean up
//	mxDestroyArray( mxParticles ) ;
//}

//void writeLogMat(ParticleSLAM particles,
//		measurementSet Z, ConstantVelocityState expectedPose,
//		gaussianMixture expectedMap, int t)
//{
//	writeParticlesMat(particles,t) ;

////	fstream zFile("measurements.log", fstream::out|fstream::app ) ;
////	for ( unsigned int n = 0 ; n < Z.size() ; n++ )
////	{
////		zFile << Z[n].range << " " << Z[n].bearing << " ";
////	}
////	zFile << endl ;
////	zFile.close() ;
////

//	// create the filename
//	std::ostringstream oss ;
//	oss << "expectation" << t << ".mat" ;
//	std::string expectationFilename = oss.str() ;

//	const char* fieldNames[] = {"pose","map"} ;
//	mxArray* mxStates = mxCreateStructMatrix(1,1,2,fieldNames) ;

////	// create the expected state mat file if it doesn't exist
////	MATFile* expectationFile ;
////	if ( !(expectationFile = matOpen("expectation.mat", "u" )) )
////		expectationFile = matOpen("expectation.mat", "w" ) ;
////
////	// create the structure of expected states if it doesn't exist
////	mxArray* mxStates ;
////	if ( !(mxStates = matGetVariable( expectationFile, "expectation" ) ) )
////	{
////		const char* fieldNames[] = {"pose","map"} ;
////		mxStates = mxCreateStructMatrix(1,1,2,fieldNames) ;
////	}

//	// pack data into mxArrays
//	mxArray* mxPose = mxCreateNumericMatrix(6,1,mxDOUBLE_CLASS,mxREAL) ;
//	mxGetPr( mxPose )[0] = expectedPose.px ;
//	mxGetPr( mxPose )[1] = expectedPose.py ;
//	mxGetPr( mxPose )[2] = expectedPose.ptheta ;
//	mxGetPr( mxPose )[3] = expectedPose.vx ;
//	mxGetPr( mxPose )[4] = expectedPose.vy ;
//	mxGetPr( mxPose )[5] = expectedPose.vtheta ;

//	const char* mapFieldNames[] = {"weights","means","covs"} ;
//	mxArray* mxMap = mxCreateStructMatrix(1,1,3,mapFieldNames) ;
//	int nFeatures = expectedMap.size() ;
//	mwSize covDims[3] = {2,2,2} ;
//	covDims[2] = nFeatures ;
//	mxArray* mxWeights = mxCreateNumericMatrix(1,nFeatures,mxDOUBLE_CLASS,mxREAL) ;
//	mxArray* mxMeans = mxCreateNumericMatrix(2,nFeatures,mxDOUBLE_CLASS,mxREAL) ;
//	mxArray* mxCovs = mxCreateNumericArray(3,covDims,mxDOUBLE_CLASS,mxREAL) ;
//	if ( nFeatures > 0 )
//	{
//		for ( int i = 0 ; i < nFeatures ; i++ )
//		{
//			mxGetPr( mxWeights )[i] = expectedMap[i].weight ;
//			mxGetPr( mxMeans )[2*i+0] = expectedMap[i].mean[0] ;
//			mxGetPr( mxMeans )[2*i+1] = expectedMap[i].mean[1] ;
//			mxGetPr( mxCovs )[4*i+0] = expectedMap[i].cov[0] ;
//			mxGetPr( mxCovs )[4*i+1] = expectedMap[i].cov[1] ;
//			mxGetPr( mxCovs )[4*i+2] = expectedMap[i].cov[2] ;
//			mxGetPr( mxCovs )[4*i+3] = expectedMap[i].cov[3] ;
//		}
//	}

//	mxSetFieldByNumber( mxMap, 0, 0, mxWeights ) ;
//	mxSetFieldByNumber( mxMap, 0, 1, mxMeans ) ;
//	mxSetFieldByNumber( mxMap, 0, 2, mxCovs ) ;

////	// resize the array to accommodate the new entry
////	mxSetM( mxStates, t+1 ) ;

//	// save the new entry
//	mxSetFieldByNumber( mxStates, 0, 0, mxPose ) ;
//	mxSetFieldByNumber( mxStates, 0, 1, mxMap ) ;

//	// write to the mat-file
//	MATFile* expectationFile = matOpen( expectationFilename.c_str(), "w") ;
//	matPutVariable( expectationFile, "expectation", mxStates ) ;
//	matClose( expectationFile ) ;

//	// clean up
//	mxDestroyArray( mxStates ) ;
//}

//void writeParticles(ParticleSLAM particles, const char* filename, int t = -1)
//{
//	std::string particlesFilename(filename) ;
//	if ( t >= 0 )
//	{
//		char timeStep[8] ;
//		sprintf(timeStep,"%d",t) ;
//		particlesFilename += timeStep ;
//	}
//	particlesFilename += ".log" ;
//	fstream particlesFile(particlesFilename.c_str(), fstream::out|fstream::app ) ;
//	if (!particlesFile)
//	{
//		cout << "failed to open log file" << endl ;
//		return ;
//	}
//	for ( unsigned int n = 0 ; n < particles.nParticles ; n++ )
//	{
//		particlesFile << particles.weights[n] << " "
//				<< particles.states[n].px << " "
//				<< particles.states[n].py << " "
//				<< particles.states[n].ptheta << " "
//				<< particles.states[n].vx << " "
//				<< particles.states[n].vy << " "
//				<< particles.states[n].vtheta << " " ;
//		for ( int i = 0 ; i < particles.maps[n].size() ; i++ )
//		{
//			particlesFile << particles.maps[n][i].weight << " "
//					<< particles.maps[n][i].mean[0] << " "
//					<< particles.maps[n][i].mean[1] << " "
//					<< particles.maps[n][i].cov[0] << " "
//					<< particles.maps[n][i].cov[1] << " "
//					<< particles.maps[n][i].cov[2] << " "
//					<< particles.maps[n][i].cov[3] << " " ;
//		}
//		particlesFile << endl ;
//	}
//	particlesFile.close() ;
//}

//void writeLog(ParticleSLAM particles,
//		measurementSet Z, ConstantVelocityState expectedPose,
//		gaussianMixture expectedMap, int t)
//{
//	writeParticles(particles,"particles",t) ;

//	fstream zFile("measurements.log", fstream::out|fstream::app ) ;
//	for ( unsigned int n = 0 ; n < Z.size() ; n++ )
//	{
//		zFile << Z[n].range << " " << Z[n].bearing << " ";
//	}
//	zFile << endl ;
//	zFile.close() ;

//	fstream stateFile("expectation.log", fstream::out|fstream::app ) ;
//	stateFile << expectedPose.px << " " << expectedPose.py << " "
//			<< expectedPose.ptheta << " " << expectedPose.vx << " "
//			<< expectedPose.vy << " " << expectedPose.vtheta << " " ;
//	for ( int n = 0 ; n < expectedMap.size() ; n++ )
//	{
//		stateFile << expectedMap[n].weight << " "
//				<< expectedMap[n].mean[0] << " "
//				<< expectedMap[n].mean[1] << " "
//				<< expectedMap[n].cov[0] << " "
//				<< expectedMap[n].cov[1] << " "
//				<< expectedMap[n].cov[2] << " "
//				<< expectedMap[n].cov[3] << " " ;
//	}
//	stateFile << endl ;
//	stateFile.close() ;
//}

//void loadConfig(const char* filename)
//{
//	fstream cfgFile(filename) ;
//	string line ;
//	string key ;
//	REAL val ;
//	int eqIdx ;
//	while( cfgFile.good() )
//	{
//		getline( cfgFile, line ) ;
//		eqIdx = line.find("=") ;
//		if ( eqIdx != string::npos )
//		{
//			line.replace(eqIdx,1," ") ;
//			istringstream iss(line) ;
//			iss >> key >> val ;
//			config.insert( pair<string,REAL>(key,val) ) ;
//		}
//	}
//	cfgFile.close() ;
//}

//int main(int argc, char *argv[])
//{
//#ifdef DEBUG
//	time_t rawtime ;
//	struct tm *timeinfo ;
//	time( &rawtime ) ;
//	timeinfo = localtime( &rawtime ) ;
//	strftime(timestamp, 80, "%Y%m%d-%H%M%S", timeinfo ) ;
//	mkdir(timestamp, S_IRWXU) ;
//#endif
//	std::vector<measurementSet> allMeasurements ;
//	allMeasurements = loadMeasurements() ;
//	std::vector<measurementSet>::iterator i( allMeasurements.begin() ) ;
//	std::vector<RangeBearingMeasurement>::iterator ii ;
//	int nSteps = allMeasurements.size() ;

//	// load the configuration file
//	DEBUG_MSG("Loading configuration file") ;
//	loadConfig("cfg/config.ini") ;
//#ifdef DEBUG
//	filterConfig::iterator it ;
//	for ( it = config.begin() ; it != config.end() ; it++ )
//	{
//		cout << it->first << " => " << it->second << endl ;
//	}
//#endif

//	// initialize the random number generator
//	if ( !init_sprng(DEFAULT_RNG_TYPE, make_sprng_seed(),SPRNG_DEFAULT ) )
//		cout << "Error initializing SPRNG" << endl ;

//	// initialize particles
//	ParticleSLAM particles(config["nParticles"]) ;
//	for (int n = 0 ; n < config["nParticles"] ; n++ )
//	{
//		particles.states[n].px = INITPX ;
//		particles.states[n].py = INITPY ;
//		particles.states[n].ptheta = INITPTHETA ;
//		particles.states[n].vx = INITVX ;
//		particles.states[n].vy = INITVY ;
//		particles.states[n].vtheta = INITVTHETA ;
//		particles.weights[n] = 1.0/config["nParticles"] ;
//	}

//	// check cuda device properties
//	int nDevices ;
//	cudaGetDeviceCount( &nDevices ) ;
//	cout << "Found " << nDevices << " CUDA Devices" << endl ;
//	cudaDeviceProp props ;
//	cudaGetDeviceProperties( &props, 0 ) ;
//	cout << "Device name: " << props.name << endl ;
//	cout << "Compute capability: " << props.major << "." << props.minor << endl ;
//	deviceMemLimit = props.totalGlobalMem*0.95 ;
//	cout << "Setting device memory limit to " << deviceMemLimit << " bytes" << endl ;

////	cudaPrintfInit() ;

//	// do the simulation
//	measurementSet ZZ ;
//	measurementSet ZPrev ;
//	ParticleSLAM particlesPreMerge(particles) ;
//	ConstantVelocityState expectedPose ;
//	gaussianMixture expectedMap ;
//	REAL nEff ;
//	timeval start, stop ;
//	cout << "STARTING SIMULATION" << endl ;
//	for (int n = 0 ; n < nSteps ; n++ )
//	{
//		gettimeofday( &start, NULL ) ;
//		cout << "****** Time Step [" << n << "/" << nSteps << "] ******" << endl ;
//		if ( n == 113 )
//			breakUpdate = true ;
//		else
//			breakUpdate = false ;
//		ZZ = allMeasurements[n] ;
////		cout << "Measurements for time " << n << endl ;
////		for_each( Z.begin(), Z.end(), printMeasurement ) ;
//		cout << "Performing prediction" << endl ;
//		phdPredict(&particles) ;
//		if (ZPrev.size() > 0 )
//		{
//			cout << "Adding birth terms" << endl ;
//			addBirths(&particles,ZPrev) ;
//		}
////		writeParticles(particles,"particlesPreUpdate",n) ;
//		if ( ZZ.size() > 0 )
//		{
//			cout << "Performing PHD Update" << endl ;
//			particlesPreMerge = phdUpdate(&particles, ZZ) ;
////			particlesPreMerge = particles ;
//		}
//		nEff = 0 ;
//		for ( int i = 0; i < config["nParticles"] ; i++)
//			nEff += particles.weights[i]*particles.weights[i] ;
//		nEff = 1.0/nEff/config["nParticles"] ;
//		DEBUG_VAL(nEff) ;
//		if (nEff <= config["resampleThresh"])
//		{
//			DEBUG_MSG("Resampling particles") ;
//			particles = resampleParticles(particles) ;
//		}
//		recoverSlamState(particles, &expectedPose, &expectedMap ) ;
//		ZPrev = ZZ ;
//		gettimeofday( &stop, NULL ) ;
//		double elapsed = (stop.tv_sec - start.tv_sec)*1000 ;
//		elapsed += (stop.tv_usec - start.tv_usec)/1000 ;
//		fstream timeFile("loopTime.log", fstream::out|fstream::app ) ;
//		timeFile << elapsed << endl ;
//		timeFile.close() ;
//#ifdef DEBUG
//		DEBUG_MSG( "Writing Log" ) ;
////		writeParticles(particlesPreMerge,"particlesPreMerge",n) ;
////		writeLog(particles, ZZ, expectedPose, expectedMap, n) ;
//		writeLogMat(particles, ZZ, expectedPose, expectedMap, n) ;
//#endif
//		if ( isnan(nEff) )
//		{
//			cout << "nan weights detected! exiting..." << endl ;
//			break ;
//		}
//		for ( int i =0 ; i < config["nParticles"] ; i++ )
//		{
//			for ( int j = 0 ; j < particles.maps[i].size() ; j++ )
//			{
//				if ( particles.maps[i][j].weight == 0 )
//				{
//					DEBUG_MSG("Invalid features detected!") ;
//					exit(1) ;
//				}
//			}
//		}
//	}
//#ifdef DEBUG
//	string command("mv *.mat ") ;
//	command += timestamp ;
//	system( command.c_str() ) ;
//	command = "mv *.log " ;
//	command += timestamp ;
//	system( command.c_str() ) ;
//#endif
////	cudaPrintfEnd() ;
//	cout << "DONE!" << endl ;
//	return 0 ;
//}
