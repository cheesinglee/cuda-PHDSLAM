#ifndef DEVICE_MATH_H
#define DEVICE_MATH_H

#include "slamtypes.h"
#include <float.h>

/// return the closest symmetric positve definite matrix for 2x2 input
__device__ void
makePositiveDefinite( REAL A[4] ) ;

/// compute the Mahalanobis distance between two Gaussians
__device__ REAL
computeMahalDist(Gaussian2D a, Gaussian2D b) ;


#endif // DEVICE_MATH_H
