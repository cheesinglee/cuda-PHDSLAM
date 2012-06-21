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

#define STATIC_MODEL 0
#define DYNAMIC_MODEL 1
#define MIXED_MODEL 2

#define STATIC_MEASUREMENT 0
#define DYNAMIC_MEASUREMENT 1

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
    int label ;
} ;

typedef struct{
    REAL u ;
    REAL v ;
} ImageMeasurement ;

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

    // camera stuff
    int particlesPerFeature ;
    int imageWidth ;
    int imageHeight ;
    REAL stdU ;
    REAL stdV ;
    REAL disparityBirth ;
    REAL stdDBirth ;


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
    int featureModel ;

    int motionType ;
    int mapEstimate ;
    int cphdDistType ;
    REAL nu ;
    bool labeledMeasurements ;

    // ackerman steering stuff
    REAL l ;
    REAL h ;
    REAL a ;
    REAL b ;
    REAL stdAlpha ;
    REAL stdEncoder ;

    // state export configuration
    bool saveAllMaps ;
    bool savePrediction ;
} SlamConfig ;

//typedef vector<PoseParticle> ParticleVector ;
typedef vector<Gaussian2D> GaussianMixture ;
typedef vector<RangeBearingMeasurement> measurementSet ;

class ParticleSLAM{
public:
    int n_particles ;
    vector<double> weights ;
    vector<ConstantVelocityState> states ;
    vector<int> resample_idx ;

    ParticleSLAM(unsigned int n = 100) :  n_particles(n),weights(n),
      states(n) {}
};

class SynthSLAM : public ParticleSLAM{
public:
    vector<vector<Gaussian2D> > maps_static ;
    vector<vector<Gaussian4D> > maps_dynamic ;
    vector<Gaussian2D> map_estimate_static ;
    vector<Gaussian4D> map_estimate_dynamic ;
    vector< vector<REAL> > cardinalities ;
    vector<REAL> cardinality_birth ;

    SynthSLAM(unsigned int n) : ParticleSLAM(n),
        maps_static(n),
        maps_dynamic(n),
        map_estimate_static(),
        map_estimate_dynamic(),
        cardinalities(n),
        cardinality_birth()
    {
    }
};


struct SmcPhdStatic{
    vector<REAL> x ;
    vector<REAL> y ;
};

struct SmcPhdDynamic:SmcPhdStatic{
    vector<REAL> vx ;
    vector<REAL> vy ;
};

class SmcPhdSLAM{
public:
    int n_particles ;
    vector<REAL> weights ;
    vector<ConstantVelocityState> particles ;
    vector<SmcPhdStatic> maps_static ;
    vector<SmcPhdDynamic> maps_dynamic ;
};



typedef struct{
    ConstantVelocityState pose ;
    REAL fx ;
    REAL fy ;
    REAL u0 ;
    REAL v0 ;
} CameraState ;

typedef struct{
    vector<REAL> x ;
    vector<REAL> y ;
    vector<REAL> z ;
    vector<REAL> weights ;
} ParticleMap ;

class DisparitySLAM : public ParticleSLAM{
public:
    vector<ParticleMap> maps ;
    ParticleMap map_estimate ;
    vector<CameraState> states ;

    DisparitySLAM(CameraState initial, unsigned int n) : ParticleSLAM(n), maps(n), states(n,initial) {}
};


void
disparityUpdate(DisparitySLAM& slam,
                std::vector<ImageMeasurement> measurements) ;

void
disparityPredict(DisparitySLAM& slam) ;

#endif /* SLAMTYPES_H_ */
