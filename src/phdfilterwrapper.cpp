#include "phdfilterwrapper.h"
#include "slamtypes.h"

//--- Externally defined CUDA kernel callers
extern "C"
void
phdPredict( ParticleSLAM& particles ) ;

extern "C"
void
addBirths( ParticleSLAM& particles, measurementSet ZPrev ) ;

extern "C"
ParticleSLAM
phdUpdate(ParticleSLAM& particles, measurementSet measurements) ;

extern "C"
ParticleSLAM resampleParticles( ParticleSLAM oldParticles, int n_particles_new = -1 ) ;

extern "C"
void recoverSlamState(ParticleSLAM particles, ConstantVelocityState& expectedPose,
		gaussianMixture& expectedMap) ;

extern "C"
void setDeviceConfig( const SlamConfig& config ) ;
//--- End external functions

PhdFilterWrapper::PhdFilterWrapper()
{
}
