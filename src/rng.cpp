#include "rng.h"
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <ctime>
#include <iostream>

// random number generator engine
boost::taus88 rng_g(std::time(0)) ;
boost::normal_distribution<double> normal_dist ;
boost::variate_generator< boost::taus88, boost::normal_distribution<double> > var_gen( rng_g, normal_dist ) ;
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
