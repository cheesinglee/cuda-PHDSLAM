#include "device_math.h"

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
