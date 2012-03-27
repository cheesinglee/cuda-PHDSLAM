#ifndef DEVICE_MATH_H
#define DEVICE_MATH_H

#include "slamtypes.h"
#include <float.h>

/// convolution of two 1D vectors
__host__ std::vector<REAL>
conv(std::vector<REAL> a, std::vector<REAL> b)
{
    int m = a.size() ;
    int n = b.size() ;
    int len = m + n - 1 ;
    std::vector<REAL> c(len) ;
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
__host__ __device__ REAL
wrapAngle(REAL a)
{
    REAL remainder = fmod(a, REAL(2*M_PI)) ;
    if ( remainder > M_PI )
        remainder -= 2*M_PI ;
    else if ( remainder < -M_PI )
        remainder += 2*M_PI ;
    return remainder ;
}

/// return the closest symmetric positve definite matrix for 2x2 input
__device__ void
makePositiveDefinite( REAL A[4] )
{
    // eigenvalues:
    REAL detA = A[0]*A[3] + A[1]*A[2] ;
    // check if already positive definite
    if ( detA > 0 && A[0] > 0 )
    {
        A[1] = (A[1] + A[2])/2 ;
        A[2] = A[1] ;
        return ;
    }
    REAL trA = A[0] + A[3] ;
    REAL trA2 = trA*trA ;
    REAL eval1 = 0.5*trA + 0.5*sqrt( trA2 - 4*detA ) ;
    REAL eval2 = 0.5*trA - 0.5*sqrt( trA2 - 4*detA ) ;

    // eigenvectors:
    REAL Q[4] ;
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
__device__ REAL
computeMahalDist(Gaussian2D a, Gaussian2D b)
{
    REAL innov[2] ;
    REAL sigma[4] ;
    REAL detSigma ;
    REAL sigmaInv[4] = {1,0,0,1} ;
    innov[0] = a.mean[0] - b.mean[0] ;
    innov[1] = a.mean[1] - b.mean[1] ;
//    sigma[0] = a.cov[0] + b.cov[0] ;
//    sigma[1] = a.cov[1] + b.cov[1] ;
//    sigma[2] = a.cov[2] + b.cov[2] ;
//    sigma[3] = a.cov[3] + b.cov[3] ;
//    sigma[0] = a.cov[0] ;
//    sigma[1] = a.cov[1] ;
//    sigma[2] = a.cov[2] ;
//    sigma[3] = a.cov[3] ;
    sigma[0] = b.cov[0] ;
    sigma[1] = b.cov[1] ;
    sigma[2] = b.cov[2] ;
    sigma[3] = b.cov[3] ;
    detSigma = sigma[0]*sigma[3] - sigma[1]*sigma[2] ;
//	detSigma = a.cov[0]*a.cov[3] - a.cov[1]*a.cov[2] ;
    if (detSigma > FLT_MIN)
    {
//		sigmaInv[0] = a.cov[3]/detSigma ;
//		sigmaInv[1] = -a.cov[1]/detSigma ;
//		sigmaInv[2] = -a.cov[2]/detSigma ;
//		sigmaInv[3] = a.cov[0]/detSigma ;
        sigmaInv[0] = sigma[3]/detSigma ;
        sigmaInv[1] = -sigma[1]/detSigma ;
        sigmaInv[2] = -sigma[2]/detSigma ;
        sigmaInv[3] = sigma[0]/detSigma ;
    }
    return  innov[0]*innov[0]*sigmaInv[0] +
            innov[0]*innov[1]*(sigmaInv[1]+sigmaInv[2]) +
            innov[1]*innov[1]*sigmaInv[3] ;
}

/// Compute the Hellinger distance between two Gaussians
__device__ REAL
computeHellingerDist( Gaussian2D a, Gaussian2D b)
{
    REAL innov[2] ;
    REAL sigma[4] ;
    REAL detSigma ;
    REAL sigmaInv[4] = {1,0,0,1} ;
    REAL dist ;
    innov[0] = a.mean[0] - b.mean[0] ;
    innov[1] = a.mean[1] - b.mean[1] ;
    sigma[0] = a.cov[0] + b.cov[0] ;
    sigma[1] = a.cov[1] + b.cov[1] ;
    sigma[2] = a.cov[2] + b.cov[2] ;
    sigma[3] = a.cov[3] + b.cov[3] ;
    detSigma = sigma[0]*sigma[3] - sigma[1]*sigma[2] ;
    if (detSigma > FLT_MIN)
    {
        sigmaInv[0] = sigma[3]/detSigma ;
        sigmaInv[1] = -sigma[1]/detSigma ;
        sigmaInv[2] = -sigma[2]/detSigma ;
        sigmaInv[3] = sigma[0]/detSigma ;
    }
    REAL epsilon = -0.25*
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
    detSigma = sigma[0]*sigma[3] - sigma[1]*sigma[2] ;
    dist *= sqrt(detSigma) ;
    dist = 1 - sqrt(dist)*exp(epsilon) ;
    return dist ;
}

/// a nan-safe logarithm
__device__ __host__
REAL safeLog( REAL x )
{
    if ( x <= 0 )
        return LOG0 ;
    else
        return log(x) ;
}

__device__ void
cholesky( REAL*A, REAL* L, int size)
{
    int i = size ;
    int n_elements = 0 ;
    while(i > 0)
    {
        n_elements += i ;
        i-- ;
    }

    int diag_idx = 0 ;
    int diag_inc = size ;
    L[0] = sqrt(A[0]) ;
    for ( i = 0 ; i < n_elements ; i++ )
    {
        if (i==diag_idx)
        {
            L[i] = A[i] ;
            diag_idx += diag_inc ;
            diag_inc-- ;
        }
    }
}

/// invert_matrix
/*!
  * Invert a symmetric, positive definite matrix using Cholesky Factorization,
  * via the Cholesky–Banachiewicz algorithm.
  */
__device__ void
invertMatrix( REAL* A, REAL* A_inv, int size)
{

}

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
sumByReduction( volatile REAL* sdata, REAL mySum, const unsigned int tid )
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
productByReduction( volatile REAL* sdata, REAL my_factor, const unsigned int tid )
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
maxByReduction( volatile REAL* sdata, REAL val, const unsigned int tid )
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


#endif // DEVICE_MATH_H
