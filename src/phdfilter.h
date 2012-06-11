#ifndef PHDFILTER_H
#define PHDFILTER_H

#ifdef __cplusplus

//--- Make kernel helper functions externally visible to C++ code
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
recoverSlamState(ParticleSLAM& particles, ConstantVelocityState& expectedPose,
        vector<REAL>& cn_estimate ) ;

void
setDeviceConfig( const SlamConfig& config ) ;
//--- End external declarations

#endif

#endif // PHDFILTER_H
