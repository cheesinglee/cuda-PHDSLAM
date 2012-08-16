#include "slamtypes.h"
#include <deque>
#include <algorithm>
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Cholesky"
#include "eigen3/Eigen/LU"

using namespace Eigen ;

struct GaussianX{
    float weight ;
    VectorXf mean ;
    MatrixXf cov ;

    template <typename T>
    GaussianX(T g){
        int dims = sizeof(g.mean)/sizeof(float) ;
        weight = g.weight ;
        mean = VectorXf(dims) ;
        cov = MatrixXf(dims,dims) ;
        for ( int i = 0 ; i < dims ; i++){
            mean(i) = g.mean[i] ;
            for( int j = 0 ; j < dims ; j++ ){
                cov(i,j) = g.cov[i+dims*j] ;
            }
        }
    }
};

float mahalanobisDistance(GaussianX a, GaussianX b){
    VectorXf diff = a.mean - b.mean ;
    MatrixXf sigma = 0.5*(a.cov+b.cov) ;
    LLT<MatrixXf> llt(sigma) ;
    MatrixXf L = llt.matrixL() ;
    VectorXf x = L.inverse()*diff ;
    return x.squaredNorm() ;
}

float mahalanobisDistance(Gaussian4D a, Gaussian4D b){
    // wrap data in Eigen objects
    using namespace Eigen ;
    Map<Vector4f> mean_a(a.mean) ;
    Map<Matrix4f> cov_a(a.cov) ;
    Map<Vector4f> mean_b(b.mean) ;
    Map<Matrix4f> cov_b(b.cov) ;

    Vector4f diff = mean_a - mean_b ;
    Matrix4f sigma = 0.5*(cov_a+cov_b) ;
    LLT<Matrix4f> llt(sigma) ;
    Matrix4f L = llt.matrixL() ;
    Vector4f x = L.inverse()*diff ;
    return x.squaredNorm() ;
}

bool compare_gaussians(GaussianX a, GaussianX b){ return a.weight > b.weight ; }

template<typename T>
vector<T>
reduceGaussianMixture(vector<T> gaussians_vec,
                      float min_distance){
    // get feature dimensionality
    int dims = sizeof(gaussians_vec[0].mean)/sizeof(float) ;

    // put features in an Eigen wrapper and store in a double-ended queue
    std::deque<GaussianX> gaussians_deque ;
    for ( unsigned int i = 0 ; i < gaussians_vec.size() ; i++ ){
        GaussianX g(gaussians_vec[i]) ;
        gaussians_deque.push_back(g);
    }
    gaussians_vec.clear();

    // output vector
    vector<T> gaussians_merge_vec ;

    // sort by weight
    std::sort(gaussians_deque.begin(),gaussians_deque.end(),
              compare_gaussians) ;
    std::deque<GaussianX> merge_deque ;
    while(gaussians_deque.size() > 0){
        // get the first item from the deque ;
        GaussianX max_element = gaussians_deque.front();
        gaussians_deque.pop_front();
//        merge_deque.push_back(max_element);

        // check distance between max element and all others
        std::deque<GaussianX>::iterator i = gaussians_deque.begin() ;
        merge_deque.clear();
        while (i != gaussians_deque.end()){
            GaussianX other_element = (*i) ;
            float d = mahalanobisDistance(max_element,other_element) ;
            if ( d < min_distance ){
                // if distance is low enough, add element to deque of gaussians
                // to be merged and remove from original deque
                merge_deque.push_back(other_element);
                i = gaussians_deque.erase(i) ;
            }
            else
            {
                i++ ;
            }
        }

        // merge gaussians in merge_deque
        GaussianX merge_element = max_element ;
        merge_element.mean *= max_element.weight ;
        for ( unsigned int i = 0 ; i < merge_deque.size() ; i++){
            merge_element.mean += merge_deque[i].weight*merge_deque[i].mean ;
            merge_element.weight += merge_deque[i].weight ;
        }
        merge_element.mean /= merge_element.weight ;
        VectorXf d = merge_element.mean - max_element.mean ;
        merge_element.cov = max_element.weight*
                (max_element.cov + d*d.transpose()) ;

        for ( unsigned int i = 0 ; i < merge_deque.size() ; i++){
            d = merge_element.mean - merge_deque[i].mean ;
            merge_element.cov += merge_deque[i].weight*(
                        merge_deque[i].cov + d*d.transpose()) ;
        }
        merge_element.cov /= merge_element.weight ;

        // convert the merged feature back to a non-Eigen-enabled type
        T merge_gaussian ;
        merge_gaussian.weight = merge_element.weight ;
        for ( int i = 0 ; i < dims ; i++ ){
            merge_gaussian.mean[i] = merge_element.mean(i) ;
            for(int j = 0 ; j < dims ; j++){
                merge_gaussian.cov[i+j*dims] = merge_element.cov(i,j) ;
            }
        }
        gaussians_merge_vec.push_back(merge_gaussian);
    }
    return gaussians_merge_vec ;
}

// explicit template instantiation
template vector<Gaussian2D>
reduceGaussianMixture<Gaussian2D>(vector<Gaussian2D> v, float d) ;

template vector<Gaussian4D>
reduceGaussianMixture<Gaussian4D>(vector<Gaussian4D> v, float d) ;
