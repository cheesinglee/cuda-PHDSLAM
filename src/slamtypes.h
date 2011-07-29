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

#define REAL double

using namespace std ;

typedef struct {
	REAL stdx ;
	REAL stdy ;
	REAL stdtheta ;
} ConstantVelocityProcessProps ;

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

// measurement structure
typedef struct{
	REAL range ;
	REAL bearing ;
} RangeBearingMeasurement;

// sensor properties structure
typedef struct{
	REAL maxRange ;
	REAL maxBearing ;
	REAL stdRange ;
	REAL stdBearing ;
	REAL clutterRate ;
	REAL probDetect ;
} RangeBearingSensorProps;

typedef struct{
	ConstantVelocityState state ;
	REAL weight ;
} PoseParticle ;

typedef struct {
	REAL cov[4] ;
	REAL mean[2] ;
	REAL weight ;
} Gaussian2D;

typedef struct{
    REAL* cov00 ;
    REAL* cov01 ;
    REAL* cov11 ;
    REAL* mean0 ;
    REAL* mean1 ;
    REAL* weight ;
} GaussianMixture2D ;

typedef struct{
    REAL x0 ;
    REAL y0 ;
    REAL theta0 ;
    REAL vx0 ;
    REAL vy0 ;
    REAL vtheta0 ;
    REAL ax ;
    REAL ay ;
    REAL atheta ;
    REAL n_alpha ;
    REAL n_encoder ;
    REAL dt ;
    REAL maxRange ;
    REAL maxBearing ;
    REAL stdRange ;
    REAL stdBearing ;
    REAL clutterRate ;
    REAL clutterDensity ;
    REAL pd ;
    int nParticles ;
    REAL resampleThresh ;
    REAL birthWeight ;
    bool gatedBirths ;
    REAL minExpectedFeatureWeight ;
    REAL minSeparation ;
    int maxFeatures ;
    REAL minFeatureWeight ;
    int particleWeighting ;

    // ackerman steering stuff
    REAL l ;
    REAL h ;
    REAL a ;
    REAL b ;
    REAL std_alpha ;
    REAL std_encoder ;
} SlamConfig ;


typedef vector<PoseParticle> ParticleVector ;
typedef vector<Gaussian2D> gaussianMixture ;
typedef vector<RangeBearingMeasurement> measurementSet ;
typedef map<string, REAL> filterConfig ;

class ParticleSLAM{
public:
    int nParticles ;
    vector<double> weights ;
    vector<ConstantVelocityState> states ;
    vector<gaussianMixture> maps ;
	vector<char> compatibleZ ;

	ParticleSLAM(unsigned int n = 100)
        :
          nParticles(n),
          weights(n),
          states(n),
          maps(n),
          compatibleZ()
	{
	}
	ParticleSLAM(const ParticleSLAM &ps)
	{
		nParticles = ps.nParticles ;
        states = ps.states ;
        maps = ps.maps ;
        weights = ps.weights ;
        compatibleZ = ps.compatibleZ ;
	}
	ParticleSLAM operator=(const ParticleSLAM ps)
	{
        nParticles = ps.nParticles ;
        states = ps.states ;
        maps = ps.maps ;
        weights = ps.weights ;
        compatibleZ = ps.compatibleZ ;
        return *this ;
	}
};

class ParticleSlamVp{
public:
    int n_particles ;
    vector<double> weights ;

};


#endif /* SLAMTYPES_H_ */
