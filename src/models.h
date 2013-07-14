#ifndef MODELS_H
#define MODELS_H

// Abstract base class for motion models
class MotionModel{
public:
    virtual void computeMotion(float* old_state, float* control, float* new_state)  ;
    virtual void computeNoisyMotion(float* old_state, float* control, float* new_state) ;
protected:

};

// Abstract base class for measurement models
class MeasurementModel{
public:
    virtual float* generate_measurement(float* observer_state, float* feature_state) ;
    virtual float* generate_noisy_measurement(float* observer_state, float* feature_state) ;
    virtual float* invert_measurement(float* measurement, float* observer_state) ;
protected:
    float* noise_cov ;
};

#endif // MODELS_H
