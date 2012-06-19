#include "device_math.cuh"
#include "slamtypes.h"

__device__ void
transformWorldToCamera(REAL xWorld, REAL yWorld, REAL zWorld,
                       CameraState cam,
                       REAL& xCamera, REAL& yCamera, REAL& zCamera){
    REAL c = cos(cam.pose.ptheta) ;
    REAL s = sin(cam.pose.ptheta) ;
    xCamera = xWorld*c + yWorld*s - cam.pose.px*c - cam.pose.py*s ;
    yCamera = -xWorld*s + yWorld*c + cam.pose.px*s - cam.pose.py*c ;
    zCamera = zWorld ;
}

__global__ void
transformWorldToDisparityKernel(REAL* xArray, REAL* yArray, REAL* zArray,
                             CameraState* cameraStates, int* cameraIdx,
                             REAL* uArray, REAL* vArray, REAL* dArray){

    int tid = blockIdx.x*blockDim.x + threadIdx.x ;
    CameraState cam = cameraStates[cameraIdx[tid]] ;

    REAL xWorld = xArray[tid] ;
    REAL yWorld = yArray[tid] ;
    REAL zWorld = zArray[tid] ;

    REAL xCamera = 0 ;
    REAL yCamera = 0 ;
    REAL zCamera = 0 ;

    transformWorldToCamera(xWorld,yWorld,zWorld,cam,xCamera,yCamera,zCamera) ;

    uArray[tid] = cam.u0 - cam.fx*xCamera/zCamera ;
    vArray[tid] = cam.v0 - cam.fy*yCamera/zCamera ;
    dArray[tid] = -fx/zCamera ;
}

__global__ void
fitGaussiansKernel(REAL* uArray, REAL* vArray, REAL* dArray,
                   int* offsets, int nGaussians,
                   Gaussian3D* gaussians){
    int tid = blockIdx.x*blockDim.x + threadIdx.x ;
    __shared__ REAL sdata[256] ;
    for (int i = blockIdx.x ; i < nGaussians ; i+=gridDim.x){
        int nParticles = offsets[i+1] - offsets[i] ;
        int offset = offsets[i] ;
        REAL val = 0 ;

        // compute mean u
        val = 0 ;
        for(int j = tid ; j < nParticles ; j+=blockDim.x){
            val += uArray[offsets+j] ;
        }
        sumByReduction(sdata,val,tid);
        REAL uMean = sdata[0]/nParticles ;
        __syncthreads() ;

        // compute mean v
        val = 0 ;
        for(int j = tid ; j < nParticles ; j+=blockDim.x){
            val += vArray[offsets+j] ;
        }
        sumByReduction(sdata,val,tid);
        REAL vMean = sdata[0]/nParticles ;
        __syncthreads() ;

        // compute mean d
        val = 0 ;
        for(int j = tid ; j < nParticles ; j+=blockDim.x){
            val += dArray[offsets+j] ;
        }
        sumByReduction(sdata,val,tid);
        REAL vMean = sdata[0]/nParticles ;
        __syncthreads() ;


        // write means to output
        if (tid == 0){
            gaussians[i].mean[0] = uMean ;
            gaussians[i].mean[1] = vMean ;
            gaussians[i].mean[2] = dMean ;
        }

        // covariance term u-u
        val = 0 ;
        for(int j = tid ; j < nParticles ; j+=blockDim.x){
            val += pow(uArray[offsets+j]-uMean,2) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0)
            gaussians[i].cov[0] = sdata[0]/(nParticles-1) ;
        __syncthreads() ;

        // covariance term v-v
        val = 0 ;
        for(int j = tid ; j < nParticles ; j+=blockDim.x){
            val += pow(vArray[offsets+j]-vMean,2) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0)
            gaussians[i].cov[4] = sdata[0]/(nParticles-1) ;
        __syncthreads() ;

        // covariance term d-d
        val = 0 ;
        for(int j = tid ; j < nParticles ; j+=blockDim.x){
            val += pow(dArray[offsets+j]-dMean,2) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0)
            gaussians[i].cov[8] = sdata[0]/(nParticles-1) ;
        __syncthreads() ;

        // covariance term u-v
        val = 0 ;
        for(int j = tid ; j < nParticles ; j+=blockDim.x){
            val += (uArray[offsets+j]-uMean)*(vArray[offsets+j]-vMean) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[1] = sdata[0]/(nParticles-1) ;
            gaussians[i].cov[3] = gaussians[i].cov[1] ;
        }
        __syncthreads() ;

        // covariance term u-d
        val = 0 ;
        for(int j = tid ; j < nParticles ; j+=blockDim.x){
            val += (uArray[offsets+j]-uMean)*(dArray[offsets+j]-dMean) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[2] = sdata[0]/(nParticles-1) ;
            gaussians[i].cov[6] = gaussians[i].cov[2] ;
        }
        __syncthreads() ;

        // covariance term v-d
        val = 0 ;
        for(int j = tid ; j < nParticles ; j+=blockDim.x){
            val += (vArray[offsets+j]-vMean)*(dArray[offsets+j]-dMean) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[5] = sdata[0]/(nParticles-1) ;
            gaussians[i].cov[7] = gaussians[i].cov[5] ;
        }
        __syncthreads() ;

    }
}
