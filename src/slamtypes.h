/*
 * slamtypes.h
 *
 *  Created on: Apr 5, 2011
 *      Author: cheesinglee
 */

#ifndef SLAMTYPES_H_
#define SLAMTYPES_H_

#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <stdlib.h>
#include <float.h>
#include <cmath>
//#include <eigen3/Eigen/Eigen>

#define REAL float
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

// constant velocity kinematic state
typedef struct{
    REAL px ;
    REAL py ;
    REAL pz ;
    REAL proll ;
    REAL ppitch ;
    REAL pyaw ;
    REAL vx ;
    REAL vy ;
    REAL vz ;
    REAL vroll ;
    REAL vpitch ;
    REAL vyaw ;
} ConstantVelocityState3D ;

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


typedef struct{
    bool debug ;

    // initial state
    REAL x0 ;
    REAL y0 ;
    REAL z0 ;
    REAL roll0 ;
    REAL pitch0 ;
    REAL yaw0 ;
    REAL vx0 ;
    REAL vy0 ;
    REAL vz0 ;
    REAL vroll0 ;
    REAL vpitch0 ;
    REAL vyaw0 ;

    // follow a set trajectory?
    bool followTrajectory ;

    // constant velocity process noise
    REAL ax ;
    REAL ay ;
    REAL az ;
    REAL aroll ;
    REAL apitch ;
    REAL ayaw ;
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
    REAL fx ;
    REAL fy ;
    REAL u0 ;
    REAL v0 ;

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

template <class StateType>
class FollowPathMotionModel{
public:
    vector<StateType> traj ;
    int counter ;
    FollowPathMotionModel(vector<StateType> traj_init){
        traj = traj_init ;
        counter = 0 ;
    }

    StateType compute_motion(){
        if (counter < traj.size())
            return traj[counter++] ;
        else
            return traj.back() ;
    }
};

//typedef vector<PoseParticle> ParticleVector ;
typedef vector<Gaussian2D> GaussianMixture ;
typedef vector<RangeBearingMeasurement> measurementSet ;
typedef vector<ImageMeasurement> imageMeasurementSet ;

class ParticleSLAM{
public:
    int n_particles ;
    vector<REAL> weights ;
    vector<ConstantVelocityState> states ;
    vector<int> resample_idx ;

    ParticleSLAM(unsigned int n = 100) :  n_particles(n),weights(n - log(n)),
        states(n),resample_idx(n) {}

    ParticleSLAM copy_particles(vector<int> indices) {return *this ;}
};

class SynthSLAM : public ParticleSLAM{
public:
    vector<vector<Gaussian2D> > maps_static ;
    vector<vector<Gaussian4D> > maps_dynamic ;
    vector<Gaussian2D> max_map_static ;
    vector<Gaussian4D> max_map_dynamic ;
    vector<Gaussian2D> exp_map_static ;
    vector<Gaussian4D> exp_map_dynamic ;
    vector< vector<REAL> > cardinalities ;
    vector<REAL> cardinality_birth ;
    vector<REAL> variances ;

    SynthSLAM(unsigned int n) : ParticleSLAM(n),
        maps_static(n),
        maps_dynamic(n),
        max_map_static(),
        max_map_dynamic(),
        exp_map_static(),
        exp_map_dynamic(),
        cardinalities(n),
        cardinality_birth(),
        variances(n)
    {
    }

    SynthSLAM copy_particles(vector<int> indices){
        SynthSLAM new_particles(indices.size()) ;
        new_particles.maps_static.clear();
        new_particles.maps_dynamic.clear();
        new_particles.cardinalities.clear();
        new_particles.weights.clear();
        new_particles.states.clear();
        new_particles.variances.clear();
        new_particles.n_particles = indices.size() ;
        for ( int n = 0 ; n < indices.size() ; n++ ){
            int i = indices[n] ;
            new_particles.maps_static.push_back(maps_static[i]);
            new_particles.maps_dynamic.push_back(maps_dynamic[i]);
            new_particles.cardinalities.push_back(cardinalities[i]);
            new_particles.weights.push_back(-log(new_particles.n_particles));
            new_particles.states.push_back(states[i]);
            new_particles.variances.push_back(variances[i]);
        }
        new_particles.resample_idx = indices ;
        return new_particles ;
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
    ConstantVelocityState3D pose ;
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

    /**
     * @brief print print particles to stdout in a MATLAB-friendly format
     */
    void print(){
        std::cout << "weights = [" ;
        for ( unsigned int i = 0; i < weights.size() ; i++ ){
            if (i > 0){
                std::cout << "," ;
            }
            std::cout << weights[i] ;
            if ( i > 0 && i % 16 == 0){
                std::cout << std::endl ;
            }
        }
        std::cout << "];" << std::endl ;

        std::cout << "particles = [" ;
        for ( unsigned int i = 0 ; i < x.size() ; i++ ){
            std::cout << x[i] << "," << y[i] << "," << z[i] ;
            if (i < x.size()-1 )
                std::cout << ";" << std::endl ;
        }
        std::cout << "];" ;
        std::cout << std::endl ;
    }
} ParticleMap ;

class DisparitySLAM : public ParticleSLAM{
public:
    vector<ParticleMap> maps ;
    ParticleMap map_estimate ;
    vector<CameraState> states ;

    DisparitySLAM(CameraState initial, unsigned int n) : ParticleSLAM(n), maps(n), states(n,initial) {}

    DisparitySLAM copy_particles(vector<int> indices){
        DisparitySLAM new_particles(states[0],indices.size()) ;
        new_particles.maps.clear();
        new_particles.weights.clear();
        new_particles.states.clear();
        for ( unsigned int n = 0 ; n < indices.size() ; n++ ){
            int i = indices[n] ;
            new_particles.maps.push_back(maps[i]);
            new_particles.weights.push_back(weights[i]);
            new_particles.states.push_back(states[i]);
        }
        new_particles.resample_idx = indices ;
        return new_particles ;
    }
};


void
disparityUpdate(DisparitySLAM& slam,
                std::vector<ImageMeasurement> measurements) ;

void
disparityPredict(DisparitySLAM& slam) ;

#endif /* SLAMTYPES_H_ */
