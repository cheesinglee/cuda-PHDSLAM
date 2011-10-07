#include "slamtypes.h"
#include "rng.h"
#include "munkres.cu"
#include "cuda.h"
#include "cutil_inline.h"

extern "C"
__device__ void
computePreUpdateComponents( ConstantVelocityState pose,
                            Gaussian2D feature, REAL* K,
                            REAL* cov_update, REAL* det_sigma,
                            REAL* S, REAL* feature_pd,
                            RangeBearingMeasurement* z_predict ) ;

extern "C"
__host__ __device__ REAL
wrapAngle(REAL a) ;

extern __shared__ REAL* sdata ;

__device__ void
munkres_assign( volatile double* C, int rows, int cols, volatile bool* starred,
                volatile bool* primed, volatile bool* row_covered,
                volatile bool* col_covered, volatile int *result, int m, int n)
{
    // initialize variables
    __shared__ int ii ;
    __shared__ int jj ;
    __shared__ int step ;
    int tid = m+n ;
    if ( tid == 0 )
      step = 0 ;
    if ( m == 0 )
        col_covered[n] = false ;
    if ( n == 0 )
    {
        row_covered[m] = false ;
        result[m] = -1 ;
    }
    starred[m+n*rows] = false ;
    primed[m+n*rows] = false ;
    __syncthreads() ;
    int k = 0 ;
    int Z0r = -1 ;
    int Z0c = -1 ;

    // preliminaries - with each column or row (whichever is larger), subtract
    // the minimum entry from all entries
    if ( rows >= cols )
    {
        double min_val = DBL_MAX ;
        for ( int m = 0 ; m < rows ; m++ )
        {
            if ( C[n*rows+m] < min_val)
                min_val = C[n*rows+m] ;
        }
        C[n*rows+m] -= min_val ;
        k = cols ;
    }
    else
    {
        double min_val = DBL_MAX ;
        for ( int n = 0 ; n < cols ; n++ )
        {
            if ( C[n*rows+m] < min_val)
                min_val = C[n*rows+m] ;
        }
        C[n*rows+m] -= min_val ;
        k = rows ;
    }
    __syncthreads() ;

    ////////////////////////////////////////////////////////////////////
    //
    // Main loop
    //
    /////////////////////////////////////////////////////////////////////
    int step = 1 ;
    while ( step < 6 )
    {
        __syncthreads() ;
        if ( step == 1 )
            // find a zero in C, and star it if there are no other
            // starred zeros in its row or column
        {
            int i = 0 ;
            int j = 0 ;
            for ( int idx = 0 ; idx < rows*cols ; idx++ )
            {
                if ( C[idx] == 0 )
                {
                    if ( tid == 0 )
                    {
                        ii = -1 ;
                        jj = -1 ;
                    }
                    __syncthreads() ;
                    // check for starred zero in row i
                    if ( m == i && n != j && starred[m+n*rows] )
                    {
                        atomicExch(&jj,n) ;
                    }
                    // check for starred zero in col j
                    if ( n == j && m != i && starred[m+n*rows] )
                    {
                        atomicExch(&ii,m) ;
                    }
                    __syncthreads() ;
                    if ( tid == 0 && ii == -1 && jj == -1 )
                    {
                        starred[i+j*rows] = true ;
                    }
                    __syncthreads() ;
                }
                i++ ;
                if ( i == rows )
                {
                    i = 0 ;
                    j++ ;
                }
            }
        }
        else if( step == 2 )
            // cover every column containing a 0*. If k columns are covered,
            // then we are done. Otherwise, go to step 3.
        {
            if ( starred[m+n*rows] )
                col_covered[n] |= 1 ;
            if ( tid == 0 )
            {
                int n_covered = 0 ;
                for ( int j = 0 ; j < cols ; j++ )
                {
                    if ( col_covered[j] )
                        n_covered++ ;
                }
                if ( n_covered == k )
                    step = 6 ;
                else
                    step = 3 ;
            }
        }
        else if( step == 3 )
            // Find an uncovered zero and prime it. If there is no 0* in its
            // row, go to step 4. Otherwise, cover this row, and uncover the
            // column containing the 0*. Continue until there are no uncovered
            // zeros and go to step 5.
        {
            bool done = false ;
            while ( !done )
            {
                // look for an uncovered zero
                int i = -1 ;
                int j = -1 ;
                for ( int n = 0 ; n < cols ; n++ )
                {
                    if ( col_covered[n] )
                        continue ;
                    for ( int m = 0 ; m < rows ; m++ )
                    {
                        if ( row_covered[m] )
                            continue ;
                        if ( C[n*rows+m] == 0 )
                        {
                            primed[n*rows+m] = true ;
                            i = m ;
                            j = n ;
                            break ;
                        }
                    }
                    if ( i >= 0 )
                        break ;
                }
                if ( i >= 0 )
                {
                    int starred_col = -1 ;
                    for ( int jj = 0 ; jj < cols ; jj++ )
                    {
                        if ( starred[jj*rows+i] )
                        {
                            starred_col = jj ;
                            break ;
                        }
                    }
                    if ( starred_col >= 0  )
                    {
                        row_covered[i] = true ;
                        col_covered[starred_col] = false;
                    }
                    else
                    {
                        Z0r = i ;
                        Z0c = j ;
                        step = 4 ;
                        done = true ;
                    }
                }
                else
                {
                    step = 5 ;
                    done = true ;
                }
            }
        }
        else if( step == 4 )
            // Let Z0 = the uncovered 0' from step 3, then Z1 is the 0* in the
            // same column of Z0 (if there is one), and Z2 is the 0' in the same
            // row as Z1 (there will always be one). Continue building this
            // sequence up to the 0' with no 0* in its column. Unstar each 0*
            // in the sequence, and star each 0'. Erase all primes and uncover
            // every line, then go to step 2.
        {
            node* head = (node*)malloc( sizeof(node) ) ;
            head->r = Z0r ;
            head->c = Z0r ;
            node* tail = head ;
            bool done = false;
            int i = -1 ;
            int j = -1 ;
            while (true)
            {
                j = tail->c ;
                // look for a star in the same column
                int star_row = -1 ;
                for ( i = 0 ; i < rows ; i++ )
                {
                    if ( starred[j*rows+i] )
                    {
                        star_row = i ;
                        break ;
                    }
                }
                if ( star_row == -1 )
                {
                    // if no star, stop building the list
                    break ;
                }
                else
                {
                    // add a node to the list
                    node* Z = (node*)malloc(sizeof(node)) ;
                    Z->r = star_row ;
                    Z->c = j ;
                    tail->next = Z ;
                    tail = Z ;

                    // find the prime in the same row as the star
                    i = star_row ;
                    int prime_col = -1 ;
                    for ( j = 0 ; j < cols ; j++ )
                    {
                        if ( primed[j*rows+i] )
                        {
                            prime_col = j ;
                            break ;
                        }
                    }

                    // add a node to the list
                    Z = (node*)malloc(sizeof(node)) ;
                    Z->r = i ;
                    Z->c = primed_col ;
                    tail = Z ;
                }
            }

            // traverse the list, star the primes, unstar the stars, and
            // destroy the nodes
            node* Z = head ;
            bool prime = true ;
            while (Z != tail )
            {
                i = Z->r ;
                j = Z->c ;
                if ( prime )
                {
                    starred[j*rows+i] = true ;
                    prime = false ;
                }
                else
                {
                    starred[j*rows+i] = false ;
                    prime = true ;
                }
                node* tmp = Z ;
                Z = Z->next ;
                free(tmp) ;
            }
            i = Z->r ;
            j = Z->c ;
            starred[j*rows+i] = true ;

            // remove all primes and uncover all lines
            i = 0 ;
            j = 0 ;
            for ( int n = 0 ; n < rows*cols ; n++ )
            {
                primed[n] = false ;
                row_covered[i] = false ;
                i++ ;
                if ( i == rows )
                {
                    i = 0 ;
                    j++ ;
                    col_covered[j] = false ;
                }
            }

            // go to step 2
            step = 2 ;
        }
        else if( step == 5 )
            // Let h be the smallest uncovered value in C. Add h to every
            // covered row, then subract h from every uncovered column.
            // Go to step 3.
        {
            // find the minimum uncovered entry of C
            double h = DBL_MAX ;
            int i = 0 ;
            int j = 0 ;
            for ( int n = 0 ; n < rows*cols ; n++ )
            {
                if ( !row_covered[i] && !col_covered[j] )
                    h = fmin(h,C[n]) ;
                i++ ;
                if ( i == rows )
                {
                    i = 0 ;
                    j++ ;
                }
            }

            // add h to every covered row, and subtract from every uncovered col
            i = 0 ;
            j = 0 ;
            for ( int n = 0 ; n < rows*cols ; n++ )
            {
                if ( row_covered[i] && col_covered[j] )
                    C[n] += h ;
                else if ( !row_covered[i] && !col_covered[j] )
                    C[n] -= h ;
                if ( i == rows )
                {
                    i = 0 ;
                    j++ ;
                }
            }
            step = 3 ;
        }
    }

    // construct the optimum assignment from the starred zeros

    for ( int i = 0 ; i < rows ; i++ )
    {
        int star_col = -1 ;
        for ( int j = 0 ; j < cols ; j++ )
        {
            if ( starred[j*rows+i] )
            {
                star_col = j ;
            }
        }
        result[i] = star_col ;
    }
}


void
preupdate_kernel(Gaussian2D features, RangeBearingMeasurement* Z, int n_measure,
                 int* map_offsets, double* Q, double* mean_update,
                 double* cov_update)
{
    int m = threadIdx.x ;
    int n = threadIdx.y ;
    int map_idx = blockIdx.x ;
    int offset = map_offsets[map_idx] ;
    int offset_feature = m+n*n_measure ;
    int offset_q = offset*n_measure ;
    int offset_mean = offset_q*2 + offset_feature ;
    int offset_cov = offset+cov*4 + offset_feature ;

    // compute single-object likelihoods
    RangeBearingMeasurement z = Z[m] ;
    Gaussian2D feature = features[offset+n] ;
    RangeBearingMeasurement z_predict ;
    double K[4] ;
    double p_update[4] ;
    double sigma_inv[4] ;
    double det_sigma ;
    double feature_pd ;
    computePreUpdateComponents( pose, feature, K, p_update, &det_sigma, sigma_inv,
                                &feature_pd, &z_predict ) ;
    double innov[2] ;
    innov[0] = z.range - z_predict ;
    innov[1] = wrapAngle(z.bearing - z_predict.bearing ) ;
    double dist = pow(innov[0],2)*sigma_inv[0]
            + innov[0]*innov[1]*(sigma_inv[1]+sigma_inv[2])
            + pow(innov[1],2)*sigma_inv[3] ;
    Q[offset_q+m+n*n_measure] = dist ;
    mean_update[offset_mean] = feature.mean[0] + K[0]*innov[0] + K[2]*innov[1] ;
    mean_update[offset_mean+1] = feature.mean[1] + K[1]*innov[0] + K[3]*innov[1] ;
    cov_update[offset_cov] = p_update[0] ;
    cov_update[offset_cov+1] = p_update[1] ;
    cov_update[offset_cov+2] = p_update[2] ;
    cov_update[offset_cov+3] = p_update[3] ;
}

void
jcbb_kernel( int* ic, double* Q, int* H, double* dist_H, int level, char* jc, int n_ic )
{

}

void data_association(ParticleSLAM& particles, measurementSet Z)
{
    size_t shmem_size = n_measure*n_features*sizeof(double)
            + n_measure*sizeof(int)
            + 2*n_measure*n_features*sizeof(bool)
            + n_measure*sizeof(bool)
            + n_features*sizeof(bool) ;
}

void update(FastSLAM& particles, measurementSet Z)
{
    // concatenate particles
    int n_particles = particles.nParticles ;
    gaussianMixture features_concat ;
    vector<int> assoc_concat ;
    vector<int> map_offsets(n_particles+1) ;
    int offset = 0 ;
    for ( int n = 0 ; n < n_particles ; n++ )
    {
        gaussianMixture map_n = particles.maps[n] ;
        vector<int> assoc_n = particles.assoc[n] ;
        features_concat.insert(features_concat.end(), map_n.begin(),
                               map_n.end() ) ;
        assoc_concat.insert( assoc_concat.end(),
                             assoc_n.begin(), assoc_n.end() ) ;
        map_offsets[n] = offset ;
        offset += map_n.size() ;
    }
    map_offsets[n_particles] = offset ;

    // determine in-range portion of map

}
