#include "rng.h"
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

// random number generator engine
boost::taus88 rng_g ;
boost::normal_distribution<double> normal_dist ;
boost::variate_generator< boost::taus88, boost::normal_distribution<double> > var_gen( rng_g, normal_dist ) ;
boost::uniform_01<> uni_dist ;

double randn()
{
    return var_gen() ;
}

double randu01()
{
    return uni_dist(rng_g) ;
}
