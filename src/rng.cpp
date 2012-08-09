#include "rng.h"
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
//#include <eigen3/Eigen/Dense>
//#include <eigen3/Eigen/Cholesky>
#include <ctime>
#include <iostream>

// random number generator engine
boost::mt19937 rng_g(std::time(0)) ;
boost::normal_distribution<double> normal_dist ;
boost::variate_generator< boost::mt19937, boost::normal_distribution<double> > var_gen( rng_g, normal_dist ) ;
boost::uniform_01<> uni_dist ;
//bool seeded = false ;

//void seed_rng()
//{
//    std::cout << "seeding rng" << std::endl ;
//    rng_g.seed( std::time(0) ) ;
//    seeded = true;
//}

double randn()
{
//    if (!seeded)
//        seed_rng() ;
    return var_gen() ;
}

double randu01()
{
//    if (!seeded)
//        seed_rng() ;
    return uni_dist(rng_g) ;
}

void randmvn3(double* mean, double* cov, int n, double* results){
    // compute cholesky decomposition of covariance matrix
    double L11 = sqrt(cov[0]) ;
    double L21 = cov[1]/L11 ;
    double L22 = sqrt(cov[4]-pow(L21,2)) ;
    double L31 = cov[2]/L11 ;
    double L32 = (cov[5]-L31*L21)/L22 ;
    double L33 = sqrt(cov[8] - pow(L31,2) - pow(L32,2)) ;

    // multiply uncorrelated normal random samples by decomposition to produce
    // correlated samples, and add mean
    for ( int i = 0 ; i < n ; i++ ){
        double x1 = randn() ;
        double x2 = randn() ;
        double x3 = randn() ;
        results[i] = x1*L11 + mean[0];
        results[i+n] = x1*L21 + x2*L22 + mean[1] ;
        results[i+2*n] = x1*L31 + x2*L32 + x3*L33 + mean[2] ;
    }
}
