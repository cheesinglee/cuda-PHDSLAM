#ifndef DEVICE_MATH_H
#define DEVICE_MATH_H

#include "slamtypes.h"
#include <float.h>

/// a nan-safe logarithm
__device__ __host__
float safeLog( float x )
{
    if ( x <= 0 )
        return LOG0 ;
    else
        return log(x) ;
}

/// evaluate generalized logistic function
__device__ __host__ float
logistic_function(float x, float lower, float upper, float beta, float tau)
{
    float y = (upper-lower)/(1+exp(-beta*(x-tau) ) ) ;
    return y ;
}

/// product of two 2x2 matrices
__device__ void
matmultiply2(float *A, float *B, float *X){
    X[0] = A[0]*B[0] + A[2]*B[1] ;
    X[1] = A[1]*B[0] + A[3]*B[1] ;
    X[2] = A[0]*B[2] + A[2]*B[3] ;
    X[3] = A[1]*B[2] + A[3]*B[3] ;
}

/// determinant of a 2x2 matrix
__host__ __device__ float
det2(float *A){
    return A[0]*A[3] - A[2]*A[1] ;
}

/// determinant of a 3x3 matrix
__host__ __device__ float
det3(float *A){
    return (A[0]*A[4]*A[8] + A[3]*A[7]*A[2] + A[6]*A[1]*A[5])
        - (A[0]*A[7]*A[5] + A[3]*A[1]*A[8] + A[6]*A[4]*A[2]) ;
}

/// determinant of a 4x4 matrix
__host__ __device__ float
det4(float *A)
{
    float det=0;
    det+=A[0]*((A[5]*A[10]*A[15]+A[9]*A[14]*A[7]+A[13]*A[6]*A[11])-(A[5]*A[14]*A[11]-A[9]*A[6]*A[15]-A[13]*A[10]*A[7]));
    det+=A[4]*((A[1]*A[14]*A[11]+A[9]*A[2]*A[15]+A[13]*A[10]*A[3])-(A[1]*A[10]*A[15]-A[9]*A[14]*A[3]-A[13]*A[2]*A[11]));
    det+=A[8]*((A[1]*A[6]*A[15]+A[5]*A[14]*A[3]+A[13]*A[2]*A[7])-(A[1]*A[14]*A[7]-A[5]*A[2]*A[15]-A[13]*A[6]*A[3]));
    det+=A[12]*((A[1]*A[10]*A[7]+A[5]*A[2]*A[12]+A[9]*A[10]*A[3])-(A[1]*A[10]*A[12]-A[5]*A[10]*A[3]-A[9]*A[2]*A[7]));
    return det ;
}

/// invert a 2x2 matrix
__device__ __host__ void
invert_matrix2(float *A, float *A_inv)
{
    float det = det2(A) ;
    A_inv[0] = A[3]/det ;
    A_inv[1] = -A[1]/det ;
    A_inv[2] = -A[2]/det ;
    A_inv[3] = A[0]/det ;
}

/// invert a 3x3 matrix
__device__ void
invert_matrix3(float *A, float* A_inv){
    float det = det3(A) ;
    A_inv[0] = (A[4]*A[8] - A[7]*A[5])/det ;
    A_inv[1] = (A[7]*A[2] - A[1]*A[8])/det ;
    A_inv[2] = (A[1]*A[5] - A[4]*A[2])/det ;
    A_inv[3] = (A[6]*A[5] - A[3]*A[8])/det ;
    A_inv[4] = (A[0]*A[8] - A[6]*A[2])/det ;
    A_inv[5] = (A[2]*A[3] - A[0]*A[5])/det ;
    A_inv[6] = (A[3]*A[7] - A[6]*A[4])/det ;
    A_inv[7] = (A[6]*A[1] - A[0]*A[7])/det ;
    A_inv[8] = (A[0]*A[4] - A[3]*A[1])/det ;
}

/// invert a 4x4 matrix
__device__ void
invert_matrix4( float *A, float *Ainv)
{
    Ainv[0] = (A[5] * A[15] * A[10] - A[5] * A[11] * A[14] - A[7] * A[13] * A[10] + A[11] * A[6] * A[13] - A[15] * A[6] * A[9] + A[7] * A[9] * A[14]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[1] = -(A[15] * A[10] * A[1] - A[11] * A[14] * A[1] + A[3] * A[9] * A[14] - A[15] * A[2] * A[9] - A[3] * A[13] * A[10] + A[11] * A[2] * A[13]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[2] = (A[5] * A[3] * A[14] - A[5] * A[15] * A[2] + A[15] * A[6] * A[1] + A[7] * A[13] * A[2] - A[3] * A[6] * A[13] - A[7] * A[1] * A[14]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[3] = -(A[5] * A[3] * A[10] - A[5] * A[11] * A[2] - A[3] * A[6] * A[9] - A[7] * A[1] * A[10] + A[11] * A[6] * A[1] + A[7] * A[9] * A[2]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[4] = -(A[15] * A[10] * A[4] - A[15] * A[6] * A[8] - A[7] * A[12] * A[10] - A[11] * A[14] * A[4] + A[11] * A[6] * A[12] + A[7] * A[8] * A[14]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[5] = (A[0] * A[15] * A[10] - A[0] * A[11] * A[14] + A[3] * A[8] * A[14] - A[15] * A[2] * A[8] + A[11] * A[2] * A[12] - A[3] * A[12] * A[10]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[6] = -(A[0] * A[15] * A[6] - A[0] * A[7] * A[14] - A[15] * A[2] * A[4] - A[3] * A[12] * A[6] + A[3] * A[4] * A[14] + A[7] * A[2] * A[12]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[7] = (-A[0] * A[7] * A[10] + A[0] * A[11] * A[6] + A[7] * A[2] * A[8] + A[3] * A[4] * A[10] - A[11] * A[2] * A[4] - A[3] * A[8] * A[6]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[8] = (-A[5] * A[15] * A[8] + A[5] * A[11] * A[12] + A[15] * A[4] * A[9] + A[7] * A[13] * A[8] - A[11] * A[4] * A[13] - A[7] * A[9] * A[12]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[9] = -(A[0] * A[15] * A[9] - A[0] * A[11] * A[13] - A[15] * A[1] * A[8] - A[3] * A[12] * A[9] + A[11] * A[1] * A[12] + A[3] * A[8] * A[13]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[10] = (A[15] * A[0] * A[5] - A[15] * A[1] * A[4] - A[3] * A[12] * A[5] - A[7] * A[0] * A[13] + A[7] * A[1] * A[12] + A[3] * A[4] * A[13]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[11] = -(A[11] * A[0] * A[5] - A[11] * A[1] * A[4] - A[3] * A[8] * A[5] - A[7] * A[0] * A[9] + A[7] * A[1] * A[8] + A[3] * A[4] * A[9]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[12] = -(-A[5] * A[8] * A[14] + A[5] * A[12] * A[10] - A[12] * A[6] * A[9] - A[4] * A[13] * A[10] + A[8] * A[6] * A[13] + A[4] * A[9] * A[14]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[13] = (-A[0] * A[13] * A[10] + A[0] * A[9] * A[14] + A[13] * A[2] * A[8] + A[1] * A[12] * A[10] - A[9] * A[2] * A[12] - A[1] * A[8] * A[14]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[14] = -(A[14] * A[0] * A[5] - A[14] * A[1] * A[4] - A[2] * A[12] * A[5] - A[6] * A[0] * A[13] + A[6] * A[1] * A[12] + A[2] * A[4] * A[13]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[15] = 0.1e1 / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]) * (A[10] * A[0] * A[5] - A[10] * A[1] * A[4] - A[2] * A[8] * A[5] - A[6] * A[0] * A[9] + A[6] * A[1] * A[8] + A[2] * A[4] * A[9]);
}

__device__ float
evalGaussian(Gaussian2D g, float* p){
    // distance from mean
    float d[2] ;
    d[0] = g.mean[0] - p[0] ;
    d[1] = g.mean[1] - p[1] ;

    // inverse covariance matrix
    float S_inv[4] ;
    invert_matrix2(g.cov,S_inv);

    // determinant of covariance matrix
    float det_S = det2(g.cov) ;

    // compute exponential
    float exponent = 0.5*(d[0]*d[0]*S_inv[0]
            + d[0]*d[1]*(S_inv[1]+S_inv[2])
            + d[1]*d[1]*S_inv[3]) ;

    return exp(exponent)/sqrt(det_S)/(2*M_PI)*g.weight ;
}

__device__ float
evalLogGaussian(Gaussian2D g, float*p){
    // distance from mean
    float d[2] ;
    d[0] = g.mean[0] - p[0] ;
    d[1] = g.mean[1] - p[1] ;

    // inverse covariance matrix
    float S_inv[4] ;
    invert_matrix2(g.cov,S_inv);

    // determinant of covariance matrix
    float det_S = det2(g.cov) ;

    // compute exponential
    float exponent = 0.5*(d[0]*d[0]*S_inv[0]
            + d[0]*d[1]*(S_inv[1]+S_inv[2])
            + d[1]*d[1]*S_inv[3]) ;

    return exponent - safeLog(sqrt(det_S)) - safeLog(2*M_PI) +
                safeLog(g.weight) ;
}

__host__ float
evalGaussianMixture(vector<Gaussian2D> gm, float* x){
    float result = 0 ;
    vector<float> d(2) ;
    float S_inv[4] ;
    for (int n = 0 ; n < gm.size() ; n++){
        d[0] = gm[n].mean[0] - x[0] ;
        d[1] = gm[n].mean[1] - x[1] ;
        invert_matrix2(gm[n].cov,S_inv);
        float det_S = det2(gm[n].cov) ;
        float exponent = 0.5*(d[0]*d[0]*S_inv[0]
                + d[0]*d[1]*(S_inv[1]+S_inv[2])
                + d[1]*d[1]*S_inv[3]) ;
        result += exp(exponent)/sqrt(det_S)/(2*M_PI) ;
    }
    return result ;
}

template <class GaussianType>
__host__ float
sumGaussianMixtureWeights(vector<GaussianType> gm){
    float weight_sum = 0 ;
    for ( int n = 0 ; n < gm.size() ; n++){
        weight_sum += gm[n].weight ;
    }
    return weight_sum ;
}

template<class GaussianType>
__device__ __host__ int
getGaussianDim(GaussianType g)
{
    int dims = sizeof(g.mean)/sizeof(float) ;
    return dims ;
}

template<class GaussianType>
__device__ __host__ GaussianType
sumGaussians(GaussianType a, GaussianType b)
{
    GaussianType result ;
    int dims = getGaussianDim(a) ;
    for (int i = 0 ; i < dims*dims ; i++ )
    {
        if (i < dims)
            result.mean[i] = a.mean[i] + b.mean[i] ;
        result.cov[i] = a.cov[i] + b.cov[i] ;
    }
    result.weight = a.weight + b.weight ;
    return result ;
}

template<class GaussianType>
__device__ __host__ void
clearGaussian(GaussianType &a)
{
    int dims = getGaussianDim(a) ;
    a.weight = 0 ;
    for (int i = 0 ; i < dims*dims ; i++)
    {
        if (i < dims)
            a.mean[i] = 0 ;
        a.cov[i] = 0 ;
    }
}

/// convolution of two 1D vectors
__host__ std::vector<float>
conv(std::vector<float> a, std::vector<float> b)
{
    int m = a.size() ;
    int n = b.size() ;
    int len = m + n - 1 ;
    std::vector<float> c(len) ;
    std::fill( c.begin(),c.end(),0) ;
    for ( int k = 0 ; k < len ; k++ )
    {
        int start_idx = max(0,k-n+1) ;
        int stop_idx = min(k,m-1) ;
        for (int j = start_idx ; j <= stop_idx ; j++ )
        {
            c[k] += a[j]*b[k-j] ;
        }
    }
    return c ;
}


/// wrap an angular value to the range [-pi,pi]
__host__ __device__ float
wrapAngle(float a)
{
    float remainder = fmod(a, float(2*M_PI)) ;
    if ( remainder > M_PI )
        remainder -= 2*M_PI ;
    else if ( remainder < -M_PI )
        remainder += 2*M_PI ;
    return remainder ;
}

/// return the closest symmetric positve definite matrix for 2x2 input
__device__ void
makePositiveDefinite( float A[4] )
{
    // eigenvalues:
    float detA = A[0]*A[3] + A[1]*A[2] ;
    // check if already positive definite
    if ( detA > 0 && A[0] > 0 )
    {
        A[1] = (A[1] + A[2])/2 ;
        A[2] = A[1] ;
        return ;
    }
    float trA = A[0] + A[3] ;
    float trA2 = trA*trA ;
    float eval1 = 0.5*trA + 0.5*sqrt( trA2 - 4*detA ) ;
    float eval2 = 0.5*trA - 0.5*sqrt( trA2 - 4*detA ) ;

    // eigenvectors:
    float Q[4] ;
    if ( fabs(A[1]) > 0 )
    {
        Q[0] = eval1 - A[3] ;
        Q[1] = A[1] ;
        Q[2] = eval2 - A[3] ;
        Q[3] = A[1] ;
    }
    else if ( fabs(A[2]) > 0 )
    {
        Q[0] = A[2] ;
        Q[1] = eval1 - A[0] ;
        Q[2] = A[2] ;
        Q[3] = eval2 - A[0] ;
    }
    else
    {
        Q[0] = 1 ;
        Q[1] = 0 ;
        Q[2] = 0 ;
        Q[3] = 1 ;
    }

    // make eigenvalues positive
    if ( eval1 < 0 )
        eval1 = DBL_EPSILON ;
    if ( eval2 < 0 )
        eval2 = DBL_EPSILON ;

    // compute the approximate matrix
    A[0] = Q[0]*Q[0]*eval1 + Q[2]*Q[2]*eval2 ;
    A[1] = Q[0]*eval1*Q[1] + Q[2]*eval2*Q[3] ;
    A[2] = A[1] ;
    A[3] = Q[1]*Q[1]*eval1 + Q[3]*Q[3]*eval2 ;
}

/// compute the Mahalanobis distance between two Gaussians
__device__ float
computeMahalDist(Gaussian2D a, Gaussian2D b)
{
    float dist = 0 ;
    float sigma_inv[4] ;
    float sigma[4] ;
    for (int i = 0 ; i <4 ; i++)
        sigma[i] = (a.cov[i] + b.cov[i])/2 ;
    invert_matrix2(sigma,sigma_inv);
    float innov[2] ;
    innov[0] = a.mean[0] - b.mean[0] ;
    innov[1] = a.mean[1] - b.mean[1] ;
    dist = innov[0]*innov[0]*sigma_inv[0] +
            innov[0]*innov[1]*(sigma_inv[1]+sigma_inv[2]) +
            innov[1]*innov[1]*sigma_inv[3] ;
    return dist ;
}

__device__ float
computeMahalDist(Gaussian3D a, Gaussian3D b)
{
    float dist = 0 ;
    float sigma_inv[9] ;
    float sigma[9] ;
    for (int i = 0 ; i <9 ; i++)
        sigma[i] = (a.cov[i] + b.cov[i])/2 ;
    invert_matrix3(sigma,sigma_inv);
    float innov[3] ;
    innov[0] = a.mean[0] - b.mean[0] ;
    innov[1] = a.mean[1] - b.mean[1] ;
    innov[2] = a.mean[1] - b.mean[1] ;
    dist = innov[0]*(sigma_inv[0]*innov[0] + sigma_inv[3]*innov[1] + sigma_inv[6]*innov[2])
            + innov[1]*(sigma_inv[1]*innov[0] + sigma_inv[4]*innov[1] + sigma_inv[7]*innov[2])
            + innov[2]*(sigma_inv[2]*innov[0] + sigma_inv[5]*innov[1] + sigma_inv[8]*innov[2]) ;
    return dist ;
}

__device__ float
computeMahalDist(Gaussian4D a, Gaussian4D b)
{
    float dist = 0 ;
    float sigma_inv[16] ;
    float sigma[16] ;
    for (int i = 0 ; i < 16 ; i++)
        sigma[i] = (a.cov[i] + b.cov[i])/2 ;
    invert_matrix4(sigma,sigma_inv) ;
    float innov[4] ;
    for ( int i = 0 ; i < 4 ; i++ )
        innov[i] = a.mean[i] - b.mean[i] ;
    dist = innov[0]*(sigma_inv[0]*innov[0] + sigma_inv[4]*innov[1] + sigma_inv[8]*innov[2] + sigma_inv[12]*innov[3])
            + innov[1]*(sigma_inv[1]*innov[0] + sigma_inv[5]*innov[1] + sigma_inv[9]*innov[2] + sigma_inv[13]*innov[3])
            + innov[2]*(sigma_inv[2]*innov[0] + sigma_inv[6]*innov[1] + sigma_inv[10]*innov[2] + sigma_inv[14]*innov[3])
            + innov[3]*(sigma_inv[3]*innov[0] + sigma_inv[7]*innov[1] + sigma_inv[11]*innov[2] + sigma_inv[15]*innov[3]) ;
    return dist ;
}

/// Compute the Hellinger distance between two Gaussians
template <class T>
__device__ float
computeHellingerDist(T a, T b)
{
    return 0 ;
}

__device__ float
computeHellingerDist( Gaussian2D a, Gaussian2D b)
{
    float dist = 0 ;
    float innov[2] ;
    float sigma[4] ;
    float detSigma ;
    float sigmaInv[4] = {1,0,0,1} ;
    innov[0] = a.mean[0] - b.mean[0] ;
    innov[1] = a.mean[1] - b.mean[1] ;
    sigma[0] = a.cov[0] + b.cov[0] ;
    sigma[1] = a.cov[1] + b.cov[1] ;
    sigma[2] = a.cov[2] + b.cov[2] ;
    sigma[3] = a.cov[3] + b.cov[3] ;
    detSigma = det2(sigma) ;
    if (detSigma > FLT_MIN)
    {
        sigmaInv[0] = sigma[3]/detSigma ;
        sigmaInv[1] = -sigma[1]/detSigma ;
        sigmaInv[2] = -sigma[2]/detSigma ;
        sigmaInv[3] = sigma[0]/detSigma ;
    }
    float epsilon = -0.25*
            (innov[0]*innov[0]*sigmaInv[0] +
             innov[0]*innov[1]*(sigmaInv[1]+sigmaInv[2]) +
             innov[1]*innov[1]*sigmaInv[3]) ;

    // determinant of half the sum of covariances
    detSigma /= 4 ;
    dist = 1/detSigma ;

    // product of covariances
    sigma[0] = a.cov[0]*b.cov[0] + a.cov[2]*b.cov[1] ;
    sigma[1] = a.cov[1]*b.cov[0] + a.cov[3]*b.cov[1] ;
    sigma[2] = a.cov[0]*b.cov[2] + a.cov[2]*b.cov[3] ;
    sigma[3] = a.cov[1]*b.cov[2] + a.cov[3]*b.cov[3] ;
    detSigma = det2(sigma) ;
    dist *= sqrt(detSigma) ;
    dist = 1 - sqrt(dist)*exp(epsilon) ;
    return dist ;
}


//__device__ void
//cholesky( float*A, float* L, int size)
//{
//    int i = size ;
//    int n_elements = 0 ;
//    while(i > 0)
//    {
//        n_elements += i ;
//        i-- ;
//    }

//    int diag_idx = 0 ;
//    int diag_inc = size ;
//    L[0] = sqrt(A[0]) ;
//    for ( i = 0 ; i < n_elements ; i++ )
//    {
//        if (i==diag_idx)
//        {
//            L[i] = A[i] ;
//            diag_idx += diag_inc ;
//            diag_inc-- ;
//        }
//    }
//}


/// device function for summations by parallel reduction in shared memory
/*!
  * Implementation based on NVIDIA whitepaper found at:
  * http://developer.download.nvidia.com/compute/cuda/1_1/Website/projects/reduction/doc/reduction.pdf
  *
  * Result is stored in sdata[0]
  \param sdata pointer to shared memory array
  \param mySum summand loaded by the thread
  \param tid thread index
  */
__device__ void
sumByReduction( volatile float* sdata, float mySum, const unsigned int tid )
{
    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads();
    if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads();

    if (tid < 32)
    {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
        sdata[tid] = mySum = mySum + sdata[tid + 16];
        sdata[tid] = mySum = mySum + sdata[tid +  8];
        sdata[tid] = mySum = mySum + sdata[tid +  4];
        sdata[tid] = mySum = mySum + sdata[tid +  2];
        sdata[tid] = mySum = mySum + sdata[tid +  1];
    }
    __syncthreads() ;
}

/// device function for products by parallel reduction in shared memory
/*!
  * Implementation based on NVIDIA whitepaper found at:
  * http://developer.download.nvidia.com/compute/cuda/1_1/Website/projects/reduction/doc/reduction.pdf
  *
  * Result is stored in sdata[0]
  \param sdata pointer to shared memory array
  \param my_factor factor loaded by the thread
  \param tid thread index
  */
__device__ void
productByReduction( volatile float* sdata, float my_factor, const unsigned int tid )
{
    sdata[tid] = my_factor;
    __syncthreads();

    // do reduction in shared mem
    if (tid < 128) { sdata[tid] = my_factor = my_factor * sdata[tid + 128]; } __syncthreads();
    if (tid <  64) { sdata[tid] = my_factor = my_factor * sdata[tid +  64]; } __syncthreads();

    if (tid < 32)
    {
        sdata[tid] = my_factor = my_factor * sdata[tid + 32];
        sdata[tid] = my_factor = my_factor * sdata[tid + 16];
        sdata[tid] = my_factor = my_factor * sdata[tid +  8];
        sdata[tid] = my_factor = my_factor * sdata[tid +  4];
        sdata[tid] = my_factor = my_factor * sdata[tid +  2];
        sdata[tid] = my_factor = my_factor * sdata[tid +  1];
    }
    __syncthreads() ;
}

/// device function for finding max value by parallel reduction in shared memory
/*!
  * Implementation based on NVIDIA whitepaper found at:
  * http://developer.download.nvidia.com/compute/cuda/1_1/Website/projects/reduction/doc/reduction.pdf
  *
  * Result is stored in sdata[0]. Other values in the array are garbage.
  \param sdata pointer to shared memory array
  \param val value loaded by the thread
  \param tid thread index
  */
__device__ void
maxByReduction( volatile float* sdata, float val, const unsigned int tid )
{
    sdata[tid] = val ;
    __syncthreads();

    // do reduction in shared mem
    if (tid < 128) { sdata[tid] = val = fmax(sdata[tid+128],val) ; } __syncthreads();
    if (tid <  64) { sdata[tid] = val = fmax(sdata[tid+64],val) ; } __syncthreads();

    if (tid < 32)
    {
        sdata[tid] = val = fmax(sdata[tid+32],val) ;
        sdata[tid] = val = fmax(sdata[tid+16],val) ;
        sdata[tid] = val = fmax(sdata[tid+8],val) ;
        sdata[tid] = val = fmax(sdata[tid+4],val) ;
        sdata[tid] = val = fmax(sdata[tid+2],val) ;
        sdata[tid] = val = fmax(sdata[tid+1],val) ;
    }
    __syncthreads() ;
}

__device__ float
logsumexpByReduction( volatile float* sdata, float val, const unsigned int tid )
{
    maxByReduction( sdata, val, tid ) ;
    float maxval = sdata[0] ;
    __syncthreads() ;

    sumByReduction( sdata, exp(val-maxval), tid) ;
    return safeLog(sdata[0]) + maxval ;
}

template <typename T>
__host__ T
logSumExp( std::vector<T> vals ){
    T maxval = *max_element(vals.begin(),vals.end()) ;
    T sum = 0 ;
    for (int i = 0 ; i < vals.size() ; i++ ){
        sum += exp(vals[i]-maxval) ;
    }
    return safeLog(sum) + maxval ;
}


typedef struct
{
    float std_accx ;
    float std_accy ;
    __device__ Gaussian4D
    compute_prediction(Gaussian4D state_prior, float dt, float scale_x, float scale_y)
    {
        Gaussian4D state_predict ;
        // predicted weight
        state_predict.weight = state_prior.weight ;

        // predicted mean
        state_predict.mean[0] = state_prior.mean[0] + dt*state_prior.mean[2] ;
        state_predict.mean[1] = state_prior.mean[1] + dt*state_prior.mean[3] ;
        state_predict.mean[2] = state_prior.mean[2] ;
        state_predict.mean[3] = state_prior.mean[3] ;

        // predicted covariance
        float var_x = pow(std_accx,2)*scale_x ;
        float var_y = pow(std_accy,2)*scale_y ;

        state_predict.cov[0] = state_prior.cov[0] + state_prior.cov[8] * dt
                + dt * (state_prior.cov[2] + state_prior.cov[10] * dt)
                + powf(dt, 0.4e1) * var_x / 0.4e1;
        state_predict.cov[1] = state_prior.cov[1] + state_prior.cov[9] * dt
                + dt * (state_prior.cov[3] + state_prior.cov[11] * dt);
        state_predict.cov[2] = state_prior.cov[2] + state_prior.cov[10] * dt
                + powf(dt, 0.3e1) * var_x / 0.2e1;
        state_predict.cov[3] = state_prior.cov[3] + state_prior.cov[11] * dt;
        state_predict.cov[4] = state_prior.cov[4] + state_prior.cov[12] * dt
                + dt * (state_prior.cov[6] + state_prior.cov[14] * dt);
        state_predict.cov[5] = state_prior.cov[5] + state_prior.cov[13] * dt
                + dt * (state_prior.cov[7] + state_prior.cov[15] * dt)
                + powf(dt, 0.4e1) * var_y / 0.4e1;
        state_predict.cov[6] = state_prior.cov[6] + state_prior.cov[14] * dt;
        state_predict.cov[7] = state_prior.cov[7] + state_prior.cov[15] * dt
                + powf(dt, 0.3e1) * var_y / 0.2e1;
        state_predict.cov[8] = state_prior.cov[8] + state_prior.cov[10] * dt
                + powf(dt, 0.3e1) * var_x / 0.2e1;
        state_predict.cov[9] = state_prior.cov[9] + state_prior.cov[11] * dt;
        state_predict.cov[10] = state_prior.cov[10] + var_x * dt * dt;
        state_predict.cov[11] = state_prior.cov[11];
        state_predict.cov[12] = state_prior.cov[12] + state_prior.cov[14] * dt;
        state_predict.cov[13] = state_prior.cov[13] + state_prior.cov[15] * dt
                + powf(dt, 0.3e1) * var_y / 0.2e1;
        state_predict.cov[14] = state_prior.cov[14];
        state_predict.cov[15] = state_prior.cov[15] + var_y * dt * dt;

        return state_predict ;
    }
} ConstantVelocityMotionModel ;

typedef struct
{
    float std_vx ;
    float std_vy ;
    __device__ Gaussian2D
    compute_prediction(Gaussian2D state_prior, float dt)
    {
        Gaussian2D state_predict ;
        // predicted weight
        state_predict.weight = state_prior.weight ;

        // predicted mean
        state_predict.mean[0] = state_prior.mean[0] ;
        state_predict.mean[1] = state_prior.mean[1] ;

        // predicted covariance
        state_predict.cov[0] = state_prior.cov[0] + pow(std_vx*dt,2) ;
        state_predict.cov[1] = state_prior.cov[1] ;
        state_predict.cov[2] = state_prior.cov[2] ;
        state_predict.cov[3] = state_prior.cov[3] + pow(std_vy*dt,2) ;

        return state_predict ;
    }
} ConstantPositionMotionModel ;

__device__ __host__
int sub_to_idx(int row, int col, int dim)
{
    int idx = row + col*dim ;
    return idx ;
}

template<class GaussianType>
__device__ __host__
void copy_gaussians(GaussianType &src, GaussianType &dest)
{
    // determine the size of the covariance matrix
    int dims = getGaussianDim(src) ;
    // copy mean and covariance
    for (int i = 0 ; i < dims*dims ; i++ )
    {
        if ( i < dims )
            dest.mean[i] = src.mean[i] ;
        dest.cov[i] = src.cov[i] ;
    }

    // copy weight
    dest.weight = src.weight ;
}

template<class GaussianType>
__device__ __host__
void force_symmetric_covariance(GaussianType &g)
{
    int dims = getGaussianDim(g) ;
    for ( int i = 0 ; i < dims ; i++ )
    {
        for( int j = 0 ; j < i ; j++)
        {
            int idx_lower = sub_to_idx(i,j,dims) ;
            int idx_upper = sub_to_idx(j,i,dims) ;
            g.cov[idx_lower] = (g.cov[idx_lower] + g.cov[idx_upper])/2 ;
            g.cov[idx_upper] = g.cov[idx_lower] ;
        }
    }
}

// Hybrid Random Number Generator from GPU Gems 3

typedef struct{
    unsigned z1 ;
    unsigned z2 ;
    unsigned z3 ;
    unsigned z4 ;
} RngState ;

__device__
unsigned TausStep(unsigned &z, int S1, int S2, int S3, unsigned M)
{
  unsigned b=(((z << S1) ^ z) >> S2);
  return z = (((z & M) << S3) ^ b);
}

__device__
unsigned LCGStep(unsigned &z, unsigned A, unsigned C)
{
  return z=(A*z+C);
}

__device__ float
HybridTaus(RngState& state)
{
  // Combined period is lcm(p1,p2,p3,p4)~ 2^121
   return 2.3283064365387e-10 * (              // Periods
    TausStep(state.z1, 13, 19, 12, 4294967294UL) ^  // p1=2^31-1
    TausStep(state.z2, 2, 25, 4, 4294967288UL) ^    // p2=2^30-1
    TausStep(state.z3, 3, 11, 17, 4294967280UL) ^   // p3=2^28-1
    LCGStep(state.z4, 1664525, 1013904223UL)        // p4=2^32
   );
}

/// return a normally distributed random number with the box-muller transform
__device__ float2
randn(RngState& state){
    float u0=HybridTaus(state), u1=HybridTaus(state);
    float r=sqrt(-2*safeLog(u0));
    float theta=2*M_PI*u1;
    return make_float2(r*sin(theta),r*cos(theta));
}

#endif // DEVICE_MATH_H
