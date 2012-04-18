/*
 * slamtypes.h
 *
 *  Created on: Apr 5, 2011
 *      Author: cheesinglee
 */

#ifndef SLAMTYPES_H_
#define SLAMTYPES_H_

#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <stdlib.h>
#include <float.h>
//#include <eigen3/Eigen/Eigen>

#define REAL double
#define PHD_TYPE 0
#define CPHD_TYPE 1
#define CV_MOTION 0
#define ACKERMAN_MOTION 1
#define LOG0 -FLT_MAX

using namespace std ;

//typedef struct {
//	REAL stdx ;
//	REAL stdy ;
//	REAL stdtheta ;
//} ConstantVelocityProcessProps ;

// constant velocity kinematic state
typedef struct{
	REAL px ;
	REAL py ;
	REAL ptheta ;
	REAL vx ;
	REAL vy ;
	REAL vtheta ;
} ConstantVelocityState ;

// constant velocity process noise
typedef struct{
	REAL ax ;
	REAL ay ;
	REAL atheta ;
} ConstantVelocityNoise;

/// ackerman steering kinematic state
typedef struct{
    REAL px ;
    REAL py ;
    REAL ptheta ;
} AckermanState ;

/// Ackerman steering
typedef struct{
    REAL alpha ;
    REAL v_encoder ;
} AckermanControl ;

/// ackerman steering control noise
typedef struct{
    REAL n_alpha ;
    REAL n_encoder ;
} AckermanNoise;

// measurement type
class RangeBearingMeasurement {
public:
    REAL range ;
    REAL bearing ;
} ;

//// sensor properties structure
//typedef struct{
//	REAL maxRange ;
//	REAL maxBearing ;
//	REAL stdRange ;
//	REAL stdBearing ;
//	REAL clutterRate ;
//	REAL probDetect ;
//} RangeBearingSensorProps;

//typedef struct{
//	ConstantVelocityState state ;
//	REAL weight ;
//} PoseParticle ;

typedef struct {
	REAL cov[4] ;
	REAL mean[2] ;
	REAL weight ;
} Gaussian2D;

typedef struct {
    REAL cov[9] ;
    REAL mean[3] ;
    REAL weight ;
} Gaussian3D ;

typedef struct {
    REAL cov[16] ;
    REAL mean[4] ;
    REAL weight ;
} Gaussian4D ;

//typedef struct{
//    REAL* cov00 ;
//    REAL* cov01 ;
//    REAL* cov11 ;
//    REAL* mean0 ;
//    REAL* mean1 ;
//    REAL* weight ;
//} GaussianMixture2D ;

//typedef struct{
//	REAL* weights ;
//	REAL* x ;
//	REAL* y ;
//	int n_particles ;
//} ParticleMixture ;

typedef struct{
    // initial state
    REAL x0 ;
    REAL y0 ;
    REAL theta0 ;
    REAL vx0 ;
    REAL vy0 ;
    REAL vtheta0 ;

    // constant velocity process noise
    REAL ax ;
    REAL ay ;
    REAL atheta ;
    REAL dt ;
    REAL minRange ;
    REAL maxRange ;
    REAL maxBearing ;
    REAL stdRange ;
    REAL stdBearing ;
    REAL clutterRate ;
    REAL clutterDensity ;
    REAL pd ;

    // constant position process noise for targets
    REAL stdVxMap ;
    REAL stdVyMap ;

    // constant velocity process noise for targets
    REAL stdAxMap ;
    REAL stdAyMap ;

    // birth covariance for unobserved velocity terms
    REAL covVxBirth ;
    REAL covVyBirth ;

    // probability of survival
    REAL ps ;

    // jump markov parameters
    REAL tau ;
    REAL beta ;

    int n_particles ;
    int nPredictParticles ;
    int subdividePredict ;
    REAL resampleThresh ;
    REAL birthWeight ;
    REAL birthNoiseFactor ;
    bool gateBirths ;
    bool gateMeasurements ;
    REAL gateThreshold ;
    REAL minExpectedFeatureWeight ;
    REAL minSeparation ;
    int maxFeatures ;
    REAL minFeatureWeight ;
    int particleWeighting ;
    int daughterMixtureType ;
    int nDaughterParticles ;
    int maxCardinality ;
    int filterType ;
    int distanceMetric ;
    int maxSteps ;
    bool dynamicFeatures ;

    int motionType ;
    int mapEstimate ;
    int cphdDistType ;
    REAL nu ;

    // ackerman steering stuff
    REAL l ;
    REAL h ;
    REAL a ;
    REAL b ;
    REAL stdAlpha ;
    REAL stdEncoder ;
} SlamConfig ;

//typedef vector<PoseParticle> ParticleVector ;
typedef vector<Gaussian2D> GaussianMixture ;
typedef vector<RangeBearingMeasurement> measurementSet ;

class ParticleSLAM{
public:
    int n_particles ;
    vector<double> weights ;
    vector<ConstantVelocityState> states ;
    vector<vector<Gaussian2D> > maps_static ;
    vector<vector<Gaussian4D> > maps_dynamic ;
    vector< vector<REAL> > cardinalities ;
    vector<REAL> cardinality_birth ;
    vector<int> resample_idx ;

    ParticleSLAM(unsigned int n = 100)
    :
      n_particles(n),
      weights(n),
      states(n),
      maps_static(n),
      maps_dynamic(n),
      resample_idx(n),
      cardinalities(n),
      cardinality_birth()
    {
    }
    ParticleSLAM(const ParticleSLAM &ps)
    {
        n_particles = ps.n_particles ;
        states = ps.states ;
        maps_static = ps.maps_static ;
        maps_dynamic = ps.maps_dynamic ;
        weights = ps.weights ;
        cardinalities = ps.cardinalities ;
        resample_idx = ps.resample_idx ;
    }
    ParticleSLAM operator=(const ParticleSLAM ps)
    {
        n_particles = ps.n_particles ;
        states = ps.states ;
        maps_static = ps.maps_static ;
        maps_dynamic = ps.maps_dynamic ;
        weights = ps.weights ;
        cardinalities = ps.cardinalities ;
        resample_idx = ps.resample_idx ;
        return *this ;
    }
};

//class FastSLAM:ParticleSLAM{
//public:
//    vector< vector<int> > assoc ;
//    vector< MatrixX2d > likelihoods ;

//    FastSLAM( unsigned int n = 100)
//        :
//          assoc(n),
//          likelihoods(n)
//    {}

//};


#endif /* SLAMTYPES_H_ */
