#ifndef PHDFILTER_H
#define PHDFILTER_H

#ifdef __cplusplus

//--- Make kernel helper functions externally visible to C++ code
//void
//initCphdConstants() ;

void
predictMap(SynthSLAM& p) ;

void
phdPredict(SynthSLAM& particles, ... ) ;

//template<class GaussianType>
//void
//phdPredictVp( SynthSLAM& particles ) ;

SynthSLAM
phdUpdateSynth(SynthSLAM& particles, measurementSet measurements) ;

void
recoverSlamState(SynthSLAM& particles, ConstantVelocityState& expectedPose,
        vector<REAL>& cn_estimate ) ;

void
recoverSlamState(DisparitySLAM& particles, ConstantVelocityState3D& expectedPose ) ;

void
setDeviceConfig( const SlamConfig& config ) ;
//--- End external declarations

#endif

#endif // PHDFILTER_H
