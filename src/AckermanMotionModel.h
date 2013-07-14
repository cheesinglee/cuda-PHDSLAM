#ifndef ACKERMANMOTIONMODEL_H
#define ACKERMANMOTIONMODEL_H

#include "models.h"
#include <cmath>
#include <random>
#include <chrono>

float
wrapAngle(float a)
{
    float remainder = fmod(a, float(2*M_PI)) ;
    if ( remainder > M_PI )
        remainder -= 2*M_PI ;
    else if ( remainder < -M_PI )
        remainder += 2*M_PI ;
    return remainder ;
}

class AckermanMotionModel : public MotionModel{
public:
    void AckermanMotionModel(float std_alpha, float std_ve, float a, float b,
                             float l, float h){
        this->std_alpha_ = std_alpha ;
        this->std_ve_ = std_ve ;
        this->a_ = a ;
        this->b_ = b ;
        this->l_ = l ;
        this->h_ = h ;
    }

    void computeMotion(float *old_state, float *control, float *new_state){
        float x = old_state[0] ;
        float y = old_state[1] ;
        float th = old_state[2] ;

        float ve = control[0] ;
        float alpha = control[1] ;
        float dt = control[2] ;

        // transform wheel encoder velocity to vehicle-centered velocity, and
        // decompose into x, y, and theta components
        float vc = ve/(1-tan(alpha)*h_/l_) ;
        float vc_x = vc*cos(th) ;
        float vc_y = vc*sin(th) ;
        float vc_th = vc*tan(alpha)/l_ ;

        // compute new state
        new_state[0] = x + dt*(vc_x - vc_th*(a_*sin(th) + b_*cos(th))) ;
        new_state[1] = y + dt*(vc_y + vc_th*(a_*cos(th) - b_*sin(th))) ;
        new_state[2] = wrapAngle(th + dt*vc_th) ;
    }

    void computeNoisyMotion(float *old_state, float *control, float *new_state){
        float noisy_control[3] ;
        noisy_control[0] = control
    }

protected:
    // vehicle dimensions
    float a_ ; // sensor x-offset from rear axle
    float b_ ; // sensor y-offset from vehicle centerline
    float l_ ; // vehicle wheelbase length
    float h_ ; // half axle length

    // noise parameters
    float std_alpha_ ;
    float std_ve_ ;
private:
    std::m19937 rng_ ;
    std::normal_distribution randn_() ;

    void initializeRng(){
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() ;
        rng_ = std::mt19937(seed) ;
    }
};

#endif // ACKERMANMOTIONMODEL_H
