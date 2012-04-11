#ifndef PHDFILTER_H
#define PHDFILTER_H

#ifdef __cplusplus

//--- Make kernel helper functions externally visible to C++ code
void
initCphdConstants() ;

template<class GaussianType>
void
phdPredict(ParticleSLAM<GaussianType>& particles, ... ) ;

template<class GaussianType>
void
phdPredictVp( ParticleSLAM<GaussianType>& particles ) ;

template<class GaussianType>
ParticleSLAM<GaussianType>
phdUpdate(ParticleSLAM<GaussianType>& particles, measurementSet measurements) ;

template<class GaussianType>
ParticleSLAM<GaussianType>
resampleParticles( ParticleSLAM<GaussianType> oldParticles, int nParticles, vector<int>& idx_resample ) ;

template<class GaussianType>
void
recoverSlamState(ParticleSLAM<GaussianType> particles, ConstantVelocityState& expectedPose,
        vector<GaussianType>& expectedMap, vector<REAL>& cn_estimate ) ;

void
setDeviceConfig( const SlamConfig& config ) ;
//--- End external declarations

#endif

#endif // PHDFILTER_H
