#ifndef RNG_H
#define RNG_H

/*!
  This is a small library that implements a couple variate generators using
  boost_random, for use with CUDA code. It is necessary to define these in an
  external library because NVCC (as of CUDA 3.2) does not know how to handle
  the SSE intrinsics that are used in boost_random. Therefore, this code needs
  to be compiled with GCC.
 */

//extern "C"
//void seed_rng() ;

/// draws a sample from a normal distribution, with mean = 0 and standard distribution = 1
extern "C"
double randn() ;


/// draws a sample from a uniform distribution over [0,1)
extern "C"
double randu01() ;

extern "C"
void randmvn3(double* mean, double* cov, int n,double* results) ;

#endif // RNG_H
