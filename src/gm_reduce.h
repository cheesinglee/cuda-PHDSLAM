#ifndef GM_REDUCE_H
#define GM_REDUCE_H
#include <vector>

using namespace std ;

template <typename T>
vector<T> reduceGaussianMixture(vector<T> gaussians, float min_distance) ;

#endif // GM_REDUCE_H
