#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
#include <sstream>
#include <iterator>

// matlab output
#include <mex.h>
#include <mat.h>

//// pickling tools
//#include <chooseser.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <cuda.h>
#include <cutil_inline.h>

#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/type_traits/conditional.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>

#include "slamtypes.h"
#include "phdfilter.h"
#include "disparity.h"
#include "rng.h"
#include "gm_reduce.h"

//#define DEBUG

#ifdef DEBUG
#define DEBUG_MSG(x) cout << "[" << __func__ << "(" << __LINE__ << ")]: " << x << endl
#define DEBUG_VAL(x) cout << "[" << __func__ << "(" << __LINE__ << ")]: " << #x << " = " << x << endl
#else
#define DEBUG_MSG(x)
#define DEBUG_VAL(x)
#endif

using namespace std ;

// SLAM configuration
SlamConfig config ;

// device memory limit
size_t deviceMemLimit ;

// configuration file
std::string config_filename ;

// data directory
std::string data_dir ;

//// measurement datafile
//std::string measurementsFilename ;

//// control input datafile
//std::string controls_filename ;

//// measurement and control timestamp datafiles
//std::string measurements_time_filename ;
//std::string controls_time_filename ;

// time variables
time_t rawtime ;
struct tm *timeinfo ;
timeval start, stop ;
char timestamp[80] ;
REAL current_time = 0 ;
REAL last_time = 0 ;
int n_steps = -1 ;

template<class Archive>
void serialize(Archive& ar, ConstantVelocityState& s, const unsigned int version)
{
    if (version > 0 || version == 0){
        ar & s.px ;
        ar & s.py ;
        ar & s.ptheta ;
        ar & s.vx ;
        ar & s.vy ;
        ar & s.vtheta ;
    }
}

template<class Archive>
void serialize(Archive& ar, RangeBearingMeasurement& z, const unsigned int version)
{
    if (version > 0 || version == 0){
        ar & z.range ;
        ar & z.bearing ;
        ar & z.label ;
    }
}

template<class Archive>
void serialize(Archive& ar, Gaussian2D& g, const unsigned int version)
{
    if (version > 0 || version == 0){
        ar & g.weight ;
        ar & g.mean ;
        ar & g.cov ;
    }
}

template<class Archive>
void serialize(Archive& ar, Gaussian4D& g, const unsigned int version)
{
    if (version > 0 || version == 0){
        ar & g.weight ;
        ar & g.mean ;
        ar & g.cov ;
    }
}

template<class Archive>
void serialize(Archive& ar, SynthSLAM& p, const unsigned int version)
{
    if (version > 0 || version == 0){
        ar & p.cardinalities ;
        ar & p.cardinality_birth ;
        ar & p.maps_dynamic ;
        ar & p.maps_static ;
        ar & p.map_estimate_dynamic ;
        ar & p.map_estimate_static ;
        ar & p.n_particles ;
        ar & p.resample_idx ;
        ar & p.states ;
        ar & p.weights ;
    }
}

vector<REAL> loadTimestamps( string filename )
{
    fstream file( filename.c_str() ) ;
    vector<REAL> times ;
    times.clear();
    if ( file.is_open() )
    {
        while( file.good() )
        {
            string line ;
            getline(file,line) ;
            stringstream ss(line, ios_base::in ) ;
            REAL val ;
            ss >> val ;
            times.push_back( val ) ;
        }
        times.pop_back();
    }
    cout << "loaded " << times.size() << " time stamps" << endl ;
    return times ;
}

vector<AckermanControl> loadControls( string filename )
{
    string line ;
    fstream controls_file( filename.c_str() ) ;
    vector<AckermanControl> controls ;
    if ( controls_file.is_open() )
    {
        // skip header line
        getline( controls_file, line ) ;
        while( controls_file.good() )
        {
            getline( controls_file, line ) ;
            stringstream ss(line, ios_base::in ) ;
            AckermanControl u ;
            ss >> u.v_encoder >> u.alpha ;
//            cout << u.v_encoder << "\t" << u.alpha << endl ;
            controls.push_back(u) ;
        }
    }
    cout << "Loaded " << controls.size() << " control inputs" << endl ;
    return controls ;
}

void parseMeasurements(string line,measurementSet& v)
{
    string value ;
    stringstream ss(line, ios_base::in) ;

    while( ss.good() )
    {
        RangeBearingMeasurement m ;
        ss >> m.range ;
        ss >> m.bearing ;
        ss >> m.label ;

        v.push_back(m) ;
    }
//    // TODO: sloppily remove the last invalid measurement (results from newline character?)
//    v.pop_back() ;
}

void parseMeasurements(string line,imageMeasurementSet& set){
    stringstream ss(line, ios_base::in) ;
    while( ss.good() )
    {
        ImageMeasurement m ;
        ss >> m.u ;
        ss >> m.v ;
        set.push_back(m) ;
    }
}

template <typename T>
void loadMeasurements( std::string filename, vector<T>& allMeasurements )
{
    string line ;
    cout << "Opening measurements file: " << filename << endl ;
    fstream measFile(filename.c_str()) ;
    if (measFile.is_open())
    {
        // skip the header line
        getline(measFile,line) ;
        while(measFile.good())
        {
            getline(measFile,line) ;
            T measurements ;
            parseMeasurements(line,measurements);
            allMeasurements.push_back( measurements ) ;
        }
        allMeasurements.pop_back() ;
        cout << "Loaded " << allMeasurements.size() << " measurements" << endl ;
    }
    else
    {
        cout << "could not open measurements file!" << endl ;
    }
}

void loadTrajectory( std::string filename, vector<ConstantVelocityState>& trajVector ){
    cout << "Opening trajectory file: " << filename << endl ;
    fstream trajFile(filename.c_str()) ;
    if (trajFile.is_open()){
        string line ;
        while (trajFile.good()){
            getline(trajFile,line) ;
            // skip header line, if any
            if (line[0] == '%')
                continue ;
            ConstantVelocityState s ;
            stringstream ss(line,ios_base::in) ;
            ss >> s.px >> s.py >> s.ptheta >>
                  s.vx >> s.vy >> s.vtheta ;
            trajVector.push_back(s);
        }
    }
}

void loadTrajectory( std::string filename, vector<ConstantVelocityState3D>& trajVector ){
    cout << "Opening trajectory file: " << filename << endl ;
    fstream trajFile(filename.c_str()) ;
    if (trajFile.is_open()){
        string line ;
        while (trajFile.good()){
            getline(trajFile,line) ;
            // skip header line, if any
            if (line[0] == '%')
                continue ;
            ConstantVelocityState3D s ;
            stringstream ss(line,ios_base::in) ;
            ss >> s.px >> s.py >> s.pz >> s.proll >> s.ppitch >> s.pyaw >>
                  s.vx >> s.vy >> s.vz >> s.vroll >> s.vpitch >> s.vyaw ;
            trajVector.push_back(s);
        }
    }
}

void printMeasurement(RangeBearingMeasurement z)
{
    cout << z.range <<"\t\t" << z.bearing << endl ;
}

template <class GaussianType>
vector<GaussianType> computeExpectedMap(vector<vector <GaussianType> > maps,
                                        vector<REAL> weights)
// concatenate all particle maps into a single slam particle and then call the
// existing gaussian pruning algorithm ;
{
    DEBUG_MSG("Computing Expected Map") ;
    vector<GaussianType> concat ;
    int n_particles = maps.size() ;
    int total_features = 0 ;
    for ( int n = 0 ; n < n_particles ; n++ )
    {
        vector<GaussianType> map = maps[n] ;
        for ( int i = 0 ; i < map.size() ; i++ )
            map[i].weight *= exp(weights[n]) ;
        concat.insert( concat.end(), map.begin(), map.end() ) ;
        total_features += map.size() ;
    }

    if ( total_features == 0 )
    {
        DEBUG_MSG("no features") ;
        vector<GaussianType> expected_map(0) ;
        return expected_map ;
    }

    return reduceGaussianMixture(concat,config.minSeparation) ;
}

void
recoverSlamState(SynthSLAM& particles, ConstantVelocityState& expectedPose,
        vector<REAL>& cn_estimate )
{
    if ( particles.n_particles > 1 )
    {
        // calculate the weighted mean of the particle poses
        expectedPose.px = 0 ;
        expectedPose.py = 0 ;
        expectedPose.ptheta = 0 ;
        expectedPose.vx = 0 ;
        expectedPose.vy = 0 ;
        expectedPose.vtheta = 0 ;
        for ( int i = 0 ; i < particles.n_particles ; i++ )
        {
            REAL exp_weight = exp(particles.weights[i]) ;
            expectedPose.px += exp_weight*particles.states[i].px ;
            expectedPose.py += exp_weight*particles.states[i].py ;
            expectedPose.ptheta += exp_weight*particles.states[i].ptheta ;
            expectedPose.vx += exp_weight*particles.states[i].vx ;
            expectedPose.vy += exp_weight*particles.states[i].vy ;
            expectedPose.vtheta += exp_weight*particles.states[i].vtheta ;
        }

        // Maximum a priori estimate
        REAL max_weight = -FLT_MAX ;
        int max_idx = -1 ;
        for ( int i = 0 ; i < particles.n_particles ; i++ )
        {
            if ( particles.weights[i] > max_weight )
            {
                max_idx = i ;
                max_weight = particles.weights[i] ;
            }
        }
        DEBUG_VAL(max_idx) ;
//		expectedPose = particles.states[max_idx] ;

        if ( config.mapEstimate == 0)
        {
            particles.map_estimate_static = particles.maps_static[max_idx] ;
            particles.map_estimate_dynamic = particles.maps_dynamic[max_idx] ;
        }
        else
        {
            particles.map_estimate_static = computeExpectedMap(
                        particles.maps_static, particles.weights) ;
            particles.map_estimate_dynamic = computeExpectedMap(
                        particles.maps_dynamic,particles.weights) ;
        }

        cn_estimate = particles.cardinalities[max_idx] ;
    }
    else
    {
        expectedPose = particles.states[0] ;
        particles.map_estimate_static = particles.maps_static[0] ;
        particles.map_estimate_dynamic = particles.maps_dynamic[0] ;
        cn_estimate = particles.cardinalities[0] ;
    }
}

void
recoverSlamState(DisparitySLAM& particles, ConstantVelocityState3D& expectedPose)
{
    if ( particles.n_particles > 1 )
    {
        // calculate the weighted mean of the particle poses
        expectedPose.px = 0 ;
        expectedPose.py = 0 ;
        expectedPose.pz = 0 ;
        expectedPose.proll = 0 ;
        expectedPose.ppitch = 0 ;
        expectedPose.pyaw = 0 ;
        expectedPose.vx = 0 ;
        expectedPose.vy = 0 ;
        expectedPose.vz = 0 ;
        expectedPose.vroll = 0 ;
        expectedPose.vpitch = 0 ;
        expectedPose.vyaw = 0 ;
        for ( int i = 0 ; i < particles.n_particles ; i++ )
        {
            REAL exp_weight = exp(particles.weights[i]) ;
            expectedPose.px += exp_weight*particles.states[i].pose.px ;
            expectedPose.py += exp_weight*particles.states[i].pose.py ;
            expectedPose.pz += exp_weight*particles.states[i].pose.pz ;
            expectedPose.proll += exp_weight*particles.states[i].pose.proll ;
            expectedPose.ppitch += exp_weight*particles.states[i].pose.ppitch ;
            expectedPose.pyaw += exp_weight*particles.states[i].pose.pyaw ;
            expectedPose.vx += exp_weight*particles.states[i].pose.vx ;
            expectedPose.vy += exp_weight*particles.states[i].pose.vy ;
            expectedPose.vz += exp_weight*particles.states[i].pose.vz ;
            expectedPose.vroll += exp_weight*particles.states[i].pose.vroll ;
            expectedPose.vpitch += exp_weight*particles.states[i].pose.vpitch ;
            expectedPose.vyaw += exp_weight*particles.states[i].pose.vyaw ;
        }

        // Maximum a priori estimate
        REAL max_weight = -FLT_MAX ;
        int max_idx = -1 ;
        for ( int i = 0 ; i < particles.n_particles ; i++ )
        {
            if ( particles.weights[i] > max_weight )
            {
                max_idx = i ;
                max_weight = particles.weights[i] ;
            }
        }
        DEBUG_VAL(max_idx) ;

        if ( config.mapEstimate == 0)
        {
            particles.map_estimate = particles.maps[max_idx] ;
        }
        else
        {
        }
    }
    else
    {
        expectedPose = particles.states[0].pose ;
        particles.map_estimate = particles.maps[0] ;
    }
}

template <typename T>
T resampleParticles( T oldParticles, int n_new_particles)
{
    if ( n_new_particles < 0 )
    {
        n_new_particles = oldParticles.n_particles ;
    }
    vector<int> idx_resample(n_new_particles) ;
    double interval = 1.0/n_new_particles ;
    double r = randu01() * interval ;
    double c = exp(oldParticles.weights[0]) ;
    idx_resample.resize(n_new_particles, 0) ;
    int i = 0 ;
    for ( int j = 0 ; j < n_new_particles ; j++ )
    {
        r = j*interval + randu01()*interval ;
        while( r > c )
        {
            i++ ;
            // sometimes the weights don't exactly add up to 1, so i can run
            // over the indexing bounds. When this happens, find the most highly
            // weighted particle and fill the rest of the new samples with it
            if ( i >= oldParticles.n_particles || i < 0 || isnan(i) )
            {
                DEBUG_VAL(r) ;
                DEBUG_VAL(c) ;
                double max_weight = -1 ;
                int max_idx = -1 ;
                for ( int k = 0 ; k < oldParticles.n_particles ; k++ )
                {
                    DEBUG_MSG("Warning: particle weights don't add up to 1!s") ;
                    if ( exp(oldParticles.weights[k]) > max_weight )
                    {
                        max_weight = exp(oldParticles.weights[k]) ;
                        max_idx = k ;
                    }
                }
                i = max_idx ;
                // set c = 2 so that this while loop is never entered again
                c = 2 ;
                break ;
            }
            c += exp(oldParticles.weights[i]) ;
        }
        idx_resample[j] = i ;
        r += interval ;
    }
    return oldParticles.copy_particles(idx_resample) ;
}

template <class GaussianType>
void write_map_mat( vector<vector<GaussianType> > maps, mxArray*& ptr_maps)
{
    mwSize n_particles = maps.size() ;
    mxArray* ptr_weights = NULL ;
    mxArray* ptr_means = NULL ;
    mxArray* ptr_covs = NULL ;
    for ( unsigned int p = 0 ; p < n_particles ; p++ )
    {
        vector<GaussianType> map = maps[p] ;
        mwSize dims = sizeof(map[0].mean)/sizeof(REAL) ;
        mwSize map_size = map.size() ;
        mwSize cov_dims[3] = {dims,dims,map_size} ;
        ptr_weights = mxCreateNumericMatrix(1,map_size,mxDOUBLE_CLASS,mxREAL) ;
        ptr_means = mxCreateNumericMatrix(dims,map_size,mxDOUBLE_CLASS,mxREAL) ;
        ptr_covs = mxCreateNumericArray(3,cov_dims,mxDOUBLE_CLASS,mxREAL) ;
        if ( map_size > 0 )
        {
            for ( unsigned int j = 0 ; j < map_size ; j++ )
            {
                mxGetPr( ptr_weights )[j] = map[j].weight ;
                for ( unsigned int k = 0 ; k < dims ; k++ )
                {
                    mxGetPr( ptr_means )[dims*j+k] = map[j].mean[k] ;
                }
                for ( unsigned int k = 0 ; k < dims*dims ; k++ )
                {
                    mxGetPr( ptr_covs )[dims*dims*j+k] = map[j].cov[k] ;
                }
            }
        }
        mxSetFieldByNumber( ptr_maps, p, 0, ptr_weights ) ;
        mxSetFieldByNumber( ptr_maps, p, 1, ptr_means ) ;
        mxSetFieldByNumber( ptr_maps, p, 2, ptr_covs ) ;
    }
}

void write_map_mat( vector<ParticleMap> maps, mxArray*& ptr_maps)
{
    mwSize n_particles = maps.size() ;
    mxArray* ptr_particles = NULL ;
    mxArray* ptr_weights = NULL ;
    for ( unsigned int p = 0 ; p < n_particles ; p++ )
    {
        DEBUG_VAL(p) ;
        ParticleMap map = maps[p] ;
        mwSize n_features = map.x.size()/config.particlesPerFeature ;
        mwSize dims[3] = {3,config.particlesPerFeature,n_features} ;
        ptr_particles = mxCreateNumericArray(3,dims,mxDOUBLE_CLASS,mxREAL) ;
        ptr_weights = mxCreateNumericMatrix(1,n_features,mxDOUBLE_CLASS,mxREAL) ;
        DEBUG_VAL(n_features) ;
        if ( n_features > 0 )
        {
            mwSize outer_stride = config.particlesPerFeature*3 ;
            mwSize inner_stride = 3 ;
            for ( int i = 0 ; i < n_features ; i++ ){
//                DEBUG_VAL(i) ;
                for( int j = 0 ; j < config.particlesPerFeature ; j++ ){
//                    DEBUG_VAL(j) ;
                    mxGetPr(ptr_particles)[i*outer_stride+j*inner_stride] =
                            map.x[i*config.particlesPerFeature+j] ;
                    mxGetPr(ptr_particles)[i*outer_stride+j*inner_stride+1] =
                            map.y[i*config.particlesPerFeature+j] ;
                    mxGetPr(ptr_particles)[i*outer_stride+j*inner_stride+2] =
                            map.z[i*config.particlesPerFeature+j] ;
                }
                mxGetPr(ptr_weights)[i] = map.weights[i] ;
            }
            mxSetFieldByNumber(ptr_maps,p,1,ptr_weights);
            mxSetFieldByNumber(ptr_maps,p,0,ptr_particles);
        }
    }
}

void
writeParticlesMat(SynthSLAM particles, int t = -1,
                  const char* filename="particles")
{
        // create the filename
        std::ostringstream oss ;
        oss << filename ;
        oss << setfill('0') << setw(5) ;
        oss << t << ".mat" ;
        std::string matfilename = oss.str() ;

        // load particle states, weights, and resample indices into mxArrays
//        DEBUG_MSG("states,weights,resample_idx") ;
        mwSize nParticles = particles.n_particles ;
        mxArray* states = mxCreateNumericMatrix(6,nParticles,
                                                mxDOUBLE_CLASS,mxREAL) ;
        mxArray* weights = mxCreateNumericMatrix(nParticles,1,
                                                 mxDOUBLE_CLASS,mxREAL) ;
        mxArray* resample_idx = mxCreateNumericMatrix(nParticles,1,
                                                      mxINT32_CLASS,mxREAL ) ;
        int i = 0 ;
        int* ptr_resample = (int*)mxGetData(resample_idx) ;
        for ( unsigned int p = 0 ; p < nParticles ; p++ )
        {
            mxGetPr(states)[i+0] = particles.states[p].px ;
            mxGetPr(states)[i+1] = particles.states[p].py ;
            mxGetPr(states)[i+2] = particles.states[p].ptheta ;
            mxGetPr(states)[i+3] = particles.states[p].vx ;
            mxGetPr(states)[i+4] = particles.states[p].vy ;
            mxGetPr(states)[i+5] = particles.states[p].vtheta ;
            mxGetPr(weights)[p] = particles.weights[p] ;
            ptr_resample[p] = particles.resample_idx[p] ;
            i+=6 ;
        }

        // copy maps to mxarray
//        DEBUG_MSG("copy maps") ;
        const char* mapFieldNames[] = {"weights","means","covs"} ;
        mxArray* maps_static = mxCreateStructMatrix(nParticles,1,3,mapFieldNames) ;
        mxArray* maps_dynamic = mxCreateStructMatrix(nParticles,1,3,mapFieldNames) ;
        if(config.saveAllMaps)
        {
            write_map_mat( particles.maps_static, maps_static ) ;
            write_map_mat( particles.maps_dynamic, maps_dynamic ) ;
        }
        else
        {
            vector<vector<Gaussian2D> > tmp_static_map_vector ;
            vector<vector<Gaussian4D> > tmp_dynamic_map_vector ;
            tmp_static_map_vector.push_back( particles.map_estimate_static ) ;
            tmp_dynamic_map_vector.push_back( particles.map_estimate_dynamic ) ;
            write_map_mat( tmp_static_map_vector, maps_static ) ;
            write_map_mat( tmp_dynamic_map_vector, maps_dynamic ) ;
        }

        // assemble final mat-file structure
//        DEBUG_MSG("assemble mat-file") ;
        const char* particleFieldNames[] = {"states","weights","maps_static",
                                            "maps_dynamic","resample_idx"} ;
//        DEBUG_MSG("mxCreateStructMatrix") ;
        mxArray* mxParticles = mxCreateStructMatrix(1,1,5,particleFieldNames) ;
        mxSetFieldByNumber( mxParticles, 0, 0, states ) ;
        mxSetFieldByNumber( mxParticles, 0, 1, weights ) ;
        mxSetFieldByNumber( mxParticles, 0, 2, maps_static ) ;
        mxSetFieldByNumber( mxParticles, 0, 3, maps_dynamic ) ;
        mxSetFieldByNumber( mxParticles, 0, 4, resample_idx ) ;

        // write to mat file
        DEBUG_MSG("Write to mat-file") ;
        MATFile* matfile = matOpen( matfilename.c_str(), "w" ) ;
        matPutVariable( matfile, "particles", mxParticles ) ;
        matClose(matfile) ;

        // clean up
        DEBUG_MSG("mxDestroyArray") ;
        mxDestroyArray( mxParticles ) ;
}

void
writeParticlesMat(DisparitySLAM particles, int t = -1,
                  const char* filename="particles")
{
        // create the filename
        std::ostringstream oss ;
        oss << filename ;
        oss << setfill('0') << setw(5) ;
        oss << t << ".mat" ;
        std::string matfilename = oss.str() ;

        // load particle states, weights, and resample indices into mxArrays
        DEBUG_MSG("states,weights,resample_idx") ;
        mwSize nParticles = particles.n_particles ;
        mxArray* states = mxCreateNumericMatrix(12,nParticles,
                                                mxDOUBLE_CLASS,mxREAL) ;
        mxArray* weights = mxCreateNumericMatrix(nParticles,1,
                                                 mxDOUBLE_CLASS,mxREAL) ;
        mxArray* resample_idx = mxCreateNumericMatrix(nParticles,1,
                                                      mxINT32_CLASS,mxREAL ) ;
        int i = 0 ;
        int* ptr_resample = (int*)mxGetData(resample_idx) ;
        for ( unsigned int p = 0 ; p < nParticles ; p++ )
        {
            mxGetPr(states)[i+0] = particles.states[p].pose.px ;
            mxGetPr(states)[i+1] = particles.states[p].pose.py ;
            mxGetPr(states)[i+2] = particles.states[p].pose.pz ;
            mxGetPr(states)[i+3] = particles.states[p].pose.proll ;
            mxGetPr(states)[i+4] = particles.states[p].pose.ppitch ;
            mxGetPr(states)[i+5] = particles.states[p].pose.pyaw ;
            mxGetPr(states)[i+6] = particles.states[p].pose.vx ;
            mxGetPr(states)[i+7] = particles.states[p].pose.vy ;
            mxGetPr(states)[i+8] = particles.states[p].pose.vz ;
            mxGetPr(states)[i+9] = particles.states[p].pose.vroll ;
            mxGetPr(states)[i+10] = particles.states[p].pose.vpitch ;
            mxGetPr(states)[i+11] = particles.states[p].pose.vyaw ;
            mxGetPr(weights)[p] = particles.weights[p] ;
            ptr_resample[p] = particles.resample_idx[p] ;
            i+=12 ;
        }

        // copy maps to mxarray
        DEBUG_MSG("copy maps") ;
        const char* field_names[] = {"particles","weights"} ;
        mxArray* maps = mxCreateStructMatrix(nParticles,1,2,field_names) ;
        if(config.saveAllMaps)
        {
            write_map_mat( particles.maps, maps ) ;
        }
        else
        {
            maps = mxCreateStructMatrix(1,1,2,field_names) ;

            ParticleMap map = particles.map_estimate ;
            mwSize n_features = map.x.size()/config.particlesPerFeature ;
            mwSize dims[3] = {3,config.particlesPerFeature,n_features} ;
            mxArray* particles = mxCreateNumericArray(3,dims,mxDOUBLE_CLASS,mxREAL) ;
            mxArray* weights = mxCreateNumericMatrix(1,n_features,mxDOUBLE_CLASS,mxREAL) ;

            mwSize outer_stride = config.particlesPerFeature*3 ;
            mwSize inner_stride = 3 ;
            for ( int i = 0 ; i < n_features ; i++ ){
                for( int j = 0 ; j < config.particlesPerFeature ; j++ ){
                    mxGetPr(particles)[i*outer_stride+j*inner_stride] =
                            map.x[i*config.particlesPerFeature+j] ;
                    mxGetPr(particles)[i*outer_stride+j*inner_stride+1] =
                            map.y[i*config.particlesPerFeature+j] ;
                    mxGetPr(particles)[i*outer_stride+j*inner_stride+2] =
                            map.z[i*config.particlesPerFeature+j] ;
                 }
                mxGetPr(weights)[i] = map.weights[i] ;
            }
//            for ( int i = 0; i < n_features ; i++ ){
//                DEBUG_VAL(mxGetPr(weights)[i]) ;
//            }

            DEBUG_MSG("setfield particles") ;
            mxSetFieldByNumber(maps,0,0,particles);
            DEBUG_MSG("setfield weights") ;
            mxSetFieldByNumber(maps,0,1,weights);
        }

        // assemble final mat-file structure
        DEBUG_MSG("assemble mat-file") ;
        const char* particleFieldNames[] = {"states","weights","maps",
                                            "resample_idx"} ;
//        DEBUG_MSG("mxCreateStructMatrix") ;
        mxArray* mxParticles = mxCreateStructMatrix(1,1,4,particleFieldNames) ;
//        DEBUG_MSG("states") ;
        mxSetFieldByNumber( mxParticles, 0, 0, states ) ;
//        DEBUG_MSG("weights") ;
        mxSetFieldByNumber( mxParticles, 0, 1, weights ) ;
//        DEBUG_MSG("maps_static") ;
        mxSetFieldByNumber( mxParticles, 0, 2, maps ) ;
//        DEBUG_MSG("resample_idx") ;
        mxSetFieldByNumber( mxParticles, 0, 3, resample_idx ) ;

        // write to mat file
        DEBUG_MSG("Write to mat-file") ;
        MATFile* matfile = matOpen( matfilename.c_str(), "w" ) ;
        matPutVariable( matfile, "particles", mxParticles ) ;
        matClose(matfile) ;

        // clean up
        DEBUG_MSG("mxDestroyArray") ;
        mxDestroyArray( mxParticles ) ;
}

/// write a vector of features as a single line to an output stream,
/// with a terminating newline
template <class GaussianType>
void write_map(ostream out, vector<GaussianType> map)
{
    int n_features = map.size() ;
    if ( n_features > 0 )
    {
        int dims = sizeof(map[0].mean)/sizeof(REAL) ;
        for ( int n = 0 ; n < (int)map.size() ; n++ )
        {
            out << map[n].weight << " " ;
            for (int i = 0 ; i < dims ; i++ )
            {
                out << map[n].mean[i] << " " ;
            }
            for (int i = 0 ; i < dims*dims ; i++ )
            {
                out << map[n].cov[i] << " " ;
            }
        }
    }
    out << endl ;
}

void writeLog(const SynthSLAM& particles, ConstantVelocityState expectedPose,
            vector<Gaussian2D> expected_map_static, vector<Gaussian4D> expected_map_dynamic,
            vector<int> idx_resample,
                                vector<REAL> cn_estimate, int t)
{
        // create the filename
        std::ostringstream oss ;
        oss << "state_estimate" ;
        oss << setfill('0') << setw(5) ;
        oss << t << ".log" ;
        std::string filename = oss.str() ;

        fstream stateFile(filename.c_str(), fstream::out|fstream::app ) ;
        stateFile << expectedPose.px << " " << expectedPose.py << " "
                    << expectedPose.ptheta << " " << expectedPose.vx << " "
                    << expectedPose.vy << " " << expectedPose.vtheta << " "
                    << endl ;

        // write the static map
        if (expected_map_static.size() > 0)
        {
            for ( int n = 0 ; n < (int)expected_map_static.size() ; n++ )
            {
                stateFile << expected_map_static[n].weight << " " ;
                for (int i = 0 ; i < 2 ; i++ )
                {
                    stateFile << expected_map_static[n].mean[i] << " " ;
                }
                for (int i = 0 ; i < 4 ; i++ )
                {
                    stateFile << expected_map_static[n].cov[i] << " " ;
                }
            }
        }
        stateFile << endl ;

        // write the dynamic map
        if (expected_map_dynamic.size() > 0)
        {
            for ( int n = 0 ; n < (int)expected_map_dynamic.size() ; n++ )
            {
                stateFile << expected_map_dynamic[n].weight << " " ;
                for (int i = 0 ; i < 4; i++ )
                {
                    stateFile << expected_map_dynamic[n].mean[i] << " " ;
                }
                for (int i = 0 ; i < 16 ; i++ )
                {
                    stateFile << expected_map_dynamic[n].cov[i] << " " ;
                }
            }
        }
        stateFile << endl ;

        // On time step 0, there is no motion, and therefore no particle
        // shotgunning. So that every line in the log has the same
        // number of entries, we repeatedly write the particle info by the
        // shotgun factor.
        int times = 1 ;
        if ( t == 0 )
        {
            times = config.nPredictParticles ;
        }

        // particle weights
        for ( int i = 0 ; i < times; i++ )
        {
            for ( int n = 0 ; n < particles.n_particles ; n++ )
            {
                stateFile << particles.weights[n] << " " ;
            }
        }
        stateFile << endl ;

        // particle poses
        for ( int i = 0 ; i < times ; i++ )
        {
            for ( int n = 0 ; n < particles.n_particles ; n++ )
            {
                stateFile << particles.states[n].px << " "
                          << particles.states[n].py << " "
                          << particles.states[n].ptheta << " "
                          << particles.states[n].vx << " "
                          << particles.states[n].vy << " "
                          << particles.states[n].vtheta << " " ;
            }
        }
        stateFile << endl ;

        // resample indices
        for ( int n = 0 ; n < particles.n_particles ; n++ )
        {
            stateFile << idx_resample[n] << " " ;
        }
        stateFile << endl ;

        // cardinality distribution
        for ( int n = 0 ; n < config.maxCardinality+1 ; n++ )
        {
            if ( config.filterType == CPHD_TYPE )
                stateFile << cn_estimate[n] << " " ;
            else
                stateFile << "0 " ;
        }
        stateFile << endl ;
        stateFile.close() ;
}

void loadConfig(const char* filename)
{
    using namespace boost::program_options ;
    options_description desc("SLAM filter config") ;
    desc.add_options()
            ("debug", value<bool>(&config.debug)->default_value(false),"extra debug output")
            ("initial_x", value<REAL>(&config.x0)->default_value(0), "Initial x position")
            ("initial_y", value<REAL>(&config.y0)->default_value(0), "Initial y position")
            ("initial_z", value<REAL>(&config.z0)->default_value(0), "Initial z position")
            ("initial_roll", value<REAL>(&config.roll0)->default_value(0), "Initial roll")
            ("initial_pitch", value<REAL>(&config.pitch0)->default_value(0), "Initial pitch")
            ("initial_yaw", value<REAL>(&config.yaw0)->default_value(0), "Initial yaw")
            ("initial_vx", value<REAL>(&config.vx0)->default_value(0), "Initial x velocity")
            ("initial_vy", value<REAL>(&config.vy0)->default_value(0), "Initial y velocity")
            ("initial_vz", value<REAL>(&config.vy0)->default_value(0), "Initial z velocity")
            ("initial_vroll", value<REAL>(&config.vyaw0)->default_value(0), "Initial roll velocity")
            ("initial_vpitch", value<REAL>(&config.vyaw0)->default_value(0), "Initial pitch velocity")
            ("initial_vyaw", value<REAL>(&config.vyaw0)->default_value(0), "Initial yaw velocity")
            ("follow_trajectory", value<bool>(&config.followTrajectory)->default_value(false), "Follow a set trajectory")
            ("motion_type", value<int>(&config.motionType)->default_value(1), "0 = Constant Velocity, 1 = Ackerman steering")
            ("acc_x", value<REAL>(&config.ax)->default_value(0.5), "Standard deviation of x acceleration")
            ("acc_y", value<REAL>(&config.ay)->default_value(0), "Standard deviation of y acceleration")
            ("acc_z", value<REAL>(&config.az)->default_value(0), "Standard deviation of z acceleration")
            ("acc_roll", value<REAL>(&config.aroll)->default_value(0.0087), "Standard deviation of roll acceleration")
            ("acc_pitch", value<REAL>(&config.apitch)->default_value(0.0087), "Standard deviation of pitch acceleration")
            ("acc_yaw", value<REAL>(&config.ayaw)->default_value(0.0087), "Standard deviation of yaw acceleration")
            ("dt", value<REAL>(&config.dt)->default_value(0.1), "Duration of each timestep")
            ("max_bearing", value<REAL>(&config.maxBearing)->default_value(M_PI), "Maximum sensor bearing")
            ("min_range", value<REAL>(&config.minRange)->default_value(0), "Minimum sensor range")
            ("max_range", value<REAL>(&config.maxRange)->default_value(20), "Maximum sensor range")
            ("std_bearing", value<REAL>(&config.stdBearing)->default_value(0.0524), "Standard deviation of sensor bearing noise")
            ("std_range", value<REAL>(&config.stdRange)->default_value(1.0), "Standard deviation of sensor range noise")
            ("clutter_rate", value<REAL>(&config.clutterRate)->default_value(15), "Poisson mean number of clutter measurements per scan")
            ("pd", value<REAL>(&config.pd)->default_value(0.98), "Nominal probability of detection for in-range features")
            ("ps", value<REAL>(&config.ps)->default_value(0.98), "Nominal probability of survival for dynamic features")
            ("n_particles", value<int>(&config.n_particles)->default_value(512), "Number of vehicle pose particles")
			("n_predict_particles", value<int>(&config.nPredictParticles)->default_value(1), "Number of new vehicle pose particles to spawn for each prior particle when doing prediction")
            ("resample_threshold", value<REAL>(&config.resampleThresh)->default_value(0.15), "Threshold on normalized nEff for particle resampling")
            ("subdivide_predict", value<int>(&config.subdividePredict)->default_value(1), "Perform the prediction over several shorter time intervals before the update")
            ("birth_weight", value<REAL>(&config.birthWeight)->default_value(0.05), "Weight of birth features")
			("birth_noise_factor", value<REAL>(&config.birthNoiseFactor)->default_value(1.5), "Factor which multiplies the measurement noise to determine covariance of birth features")
            ("gate_births", value<bool>(&config.gateBirths)->default_value(true), "Enable measurement gating on births")
            ("gate_measurements", value<bool>(&config.gateMeasurements)->default_value(true), "Gate measurements for update")
            ("gate_threshold", value<REAL>(&config.gateThreshold)->default_value(10), "Mahalanobis distance threshold for gating")
            ("feature_model", value<int>(&config.featureModel)->default_value(0), "Feature motion model: 0-static, 1-CV, 2-static+CV")
            ("min_expected_feature_weight", value<REAL>(&config.minExpectedFeatureWeight)->default_value(0.33), "Minimum feature weight for expected map")
            ("min_separation", value<REAL>(&config.minSeparation)->default_value(5), "Minimum Mahalanobis separation between features")
            ("max_features", value<int>(&config.maxFeatures)->default_value(100), "Maximum number of features in map")
            ("min_feature_weight", value<REAL>(&config.minFeatureWeight)->default_value(0.00001), "Minimum feature weight")
            ("particle_weighting", value<int>(&config.particleWeighting)->default_value(1), "Particle weighting scheme: 1 = cluster process 2 = Vo's")
            ("daughter_mixture_type", value<int>(&config.daughterMixtureType)->default_value(0), "0: Gaussian, 1: Particle")
            ("n_daughter_particles", value<int>(&config.nDaughterParticles)->default_value(50), "Number of particles to represet each map landmark")
            ("max_cardinality", value<int>(&config.maxCardinality)->default_value(256), "Maximum cardinality for CPHD filter")
            ("filter_type", value<int>(&config.filterType)->default_value(1), "0 = PHD, 1 = CPHD")
            ("map_estimate", value<int>(&config.mapEstimate)->default_value(1), "Map state estimate 0 = MAP, 1 = EAP")
            ("cphd_disttype", value<int>(&config.cphdDistType)->default_value(0), "CPHD Cardinality distribution 0 = Binomial Poisson, 1 = COM-Poisson")
            ("nu", value<REAL>(&config.nu)->default_value(1), "COM-Poisson Parameter")
            ("distance_metric", value<int>(&config.distanceMetric)->default_value(0), "0 = Mahalanobis, 1 = Hellinger")
            ("h", value<REAL>(&config.h)->default_value(0), "Half-axle length")
            ("l", value<REAL>(&config.l)->default_value(0), "Wheelbase length")
            ("a", value<REAL>(&config.a)->default_value(0), "x-distance from rear axle to sensor")
            ("b", value<REAL>(&config.b)->default_value(0), "y-distance from centerline to sensor")
            ("std_encoder", value<REAL>(&config.stdEncoder)->default_value(0), "Std. deviation of velocity noise")
            ("std_alpha", value<REAL>(&config.stdAlpha)->default_value(0), "Std. deviation of steering angle noise")
            ("std_vx_features", value<REAL>(&config.stdVxMap)->default_value(0), "Std. deviation of x process noise in constant position model")
            ("std_vy_features", value<REAL>(&config.stdVyMap)->default_value(0), "Std. deviation of y process noise in constant position model")
            ("std_ax_features", value<REAL>(&config.stdAxMap)->default_value(0), "Std. deviation of x process noise in constant velocity model for targets")
            ("std_ay_features", value<REAL>(&config.stdAyMap)->default_value(0), "Std. deviation of y process noise in constant velocity model for targets")
            ("cov_vx_birth", value<REAL>(&config.covVxBirth)->default_value(0), "Birth covariance of x velocity in dynamic targets")
            ("cov_vy_birth", value<REAL>(&config.covVyBirth)->default_value(0), "Birth covariance of y velocity in dynamic targets")
            ("std_u", value<REAL>(&config.stdU)->default_value(1), "std deviation of measurement noise in u")
            ("std_v", value<REAL>(&config.stdV)->default_value(1), "std deviation of measurement noise in v")
            ("disparity_birth", value<REAL>(&config.disparityBirth)->default_value(1000), "birth disparity mean")
            ("image_width", value<int>(&config.imageWidth)->default_value(600), "image width in pixels")
            ("image_height", value<int>(&config.imageHeight)->default_value(480), "image width in pixels")
            ("std_d_birth", value<REAL>(&config.stdDBirth)->default_value(300), "birth std. deviation in disparity")
            ("fx", value<REAL>(&config.fx)->default_value(1000), "focal length divided by x pixel size")
            ("fy", value<REAL>(&config.fy)->default_value(1000), "focal length divided by y pixel size")
            ("u0", value<REAL>(&config.u0)->default_value(512), "principal point u coordinate")
            ("v0", value<REAL>(&config.v0)->default_value(384), "principal point v coordinate")
            ("particles_per_feature", value<int>(&config.particlesPerFeature)->default_value(100), "number of 3d particles to represent each feature")
            ("tau", value<REAL>(&config.tau)->default_value(0), "Velocity threshold for jump markov transition probability")
            ("beta", value<REAL>(&config.beta)->default_value(1), "Steepness of sigmoid function for computing JMM transition probability")
            ("labeled_measurements", value<bool>(&config.labeledMeasurements)->default_value(false), "Use static/dynamic measurement labels for computing likelihood")
            ("data_directory", value<std::string>(&data_dir)->default_value("data/"), "Path to simulation inputs")
//            ("measurements_filename", value<std::string>(&measurementsFilename)->default_value("measurements.txt"), "Path to measurements datafile")
//            ("controls_filename", value<std::string>(&controls_filename)->default_value("controls.txt"), "Path to controls datafile")
//            ("measurements_time_filename", value<std::string>(&measurements_time_filename)->default_value(""), "Path to measurement timestamps datafile")
//            ("controls_time_filename", value<std::string>(&controls_time_filename)->default_value(""), "Path to control timestamps datafile")
            ("max_time_steps", value<int>(&config.maxSteps)->default_value(10000), "Limit the number of time steps to execute")
            ("save_all_maps", value<bool>(&config.saveAllMaps)->default_value(false), "Save all particle maps")
            ("save_prediction", value<bool>(&config.savePrediction)->default_value(false), "Save the predicted state to the log files")
            ("n_steps",value<int>(&n_steps)->default_value(-1),"Number of simulation steps; if less than zero, equal to number of sensor inputs")
            ;
    ifstream ifs( filename ) ;
    if ( !ifs )
    {
        cout << "Unable to open config file: " << filename << endl ;
        cout << "Config file should contain the following options: "
                << desc << endl ;
        exit(1) ;
    }
    else
    {
        variables_map vm ;
        try{
            store( parse_config_file( ifs, desc ), vm ) ;
						notify(vm) ;
            // compute clutter density
            config.clutterDensity = config.clutterRate
                    /( 2*config.maxBearing*config.maxRange ) ;
        }
        catch( std::exception& e )
        {
            cout << "Error parsing config file: " << e.what() << endl ;
        }
    }
}

void run_synth(bool profile_run){
    cout << "running on synthetic data" << endl ;
    // load measurement data
    std::vector<measurementSet> allMeasurements ;
    std::string measurements_filename = data_dir + "measurements.txt" ;
    loadMeasurements(measurements_filename,allMeasurements) ;
    std::vector<measurementSet>::iterator i( allMeasurements.begin() ) ;
    std::vector<RangeBearingMeasurement>::iterator ii ;

    // load control inputs
    std::string controls_filename = data_dir + "controls.txt" ;
    vector<AckermanControl> all_controls ;
    all_controls = loadControls( controls_filename ) ;

    // load timestamps
    std::string measurement_times_filename = data_dir + "measurement_times.txt" ;
    vector<REAL> measurement_times = loadTimestamps( measurement_times_filename ) ;
    std::string control_times_filename = data_dir + "control_times.txt" ;
    vector<REAL> control_times = loadTimestamps( control_times_filename ) ;
    bool has_timestamps = (measurement_times.size() > 0) ;

    int nSteps = 0 ;
    if ( !has_timestamps )
    {
        nSteps = allMeasurements.size() ;
    }
    else
    {
        // check that we have the same number of timestamps as inputs
        if (measurement_times.size() != allMeasurements.size())
        {
            cout << "mismatched measurements and measurement timestamps!" << endl ;
            exit(1) ;
        }
        if ( control_times.size() != all_controls.size())
        {
            cout << "mismatched controls and controls timestamps!" << endl ;
            exit(1) ;
        }

        // maximum possible number of steps
        nSteps = measurement_times.size() + control_times.size() ;
    }
    if (nSteps > n_steps && n_steps > 0)
        nSteps = n_steps ;

    // load trajectory, if required
    vector<ConstantVelocityState> trajVector ;
    if (config.followTrajectory){
        loadTrajectory(data_dir+"traj.txt",trajVector) ;
        // only need 1 particle
        config.n_particles = 1 ;
    }

    // initialize particles
    SynthSLAM particles(config.n_particles) ;
    for (int n = 0 ; n < config.n_particles ; n++ )
    {
        particles.states[n].px = config.x0 ;
        particles.states[n].py = config.y0 ;
        particles.states[n].ptheta = config.yaw0 ;
        particles.states[n].vx = config.vx0 ;
        particles.states[n].vy = config.vy0 ;
        particles.states[n].vtheta = config.vyaw0 ;
//        particles.weights[n] = -log(config.n_particles) ;
        if ( config.filterType == CPHD_TYPE )
        {
            particles.cardinalities[n].assign( config.maxCardinality+1, -log(config.maxCardinality+1) ) ;
        }
    }
    particles.weights.assign(config.n_particles,-log(float(config.n_particles)));

//    if ( config.filterType == CPHD_TYPE )
//    {
//            particles.cardinality_birth.assign( config.maxCardinality+1, LOG0 ) ;
//            particles.cardinality_birth[0] = 0 ;
//    }
    // do the simulation
    measurementSet ZZ ;
    measurementSet ZPrev ;
    SynthSLAM particlesPreMerge(particles) ;
    ConstantVelocityState expectedPose ;
    vector<Gaussian2D> expected_map_static ;
    vector<Gaussian4D> expected_map_dynamic ;
    vector<REAL> cn_estimate ;
    REAL nEff ;
    REAL dt = 0 ;
    int z_idx = 0 ;
    int c_idx = 0 ;
    AckermanControl current_control ;
    current_control.alpha = 0 ;
    current_control.v_encoder = 0 ;
    bool do_predict = false ;

    if(!profile_run)
    {
    cout << "STARTING SIMULATION" << endl ;
//    if ( config.filterType == CPHD_TYPE )
//    {
//        DEBUG_MSG("Initializing CPHD constants") ;
//        initCphdConstants() ;
//    }

    for (int n = 0 ; n < nSteps ; n++ )
    {
        gettimeofday( &start, NULL ) ;
        cout << "****** Time Step [" << n << "/" << nSteps << "] ******" << endl ;
        time( &rawtime ) ;
        timeinfo = localtime( &rawtime ) ;
        strftime(timestamp, 80, "%Y%m%d-%H%M%S", timeinfo ) ;
        cout << timestamp << endl ;
        // get inputs for this time step
        if ( has_timestamps )
        {
            if (z_idx >= measurement_times.size() || c_idx >= control_times.size()){
                DEBUG_MSG("no more timestamps") ;
                break ;
            }
            if (measurement_times[z_idx] < control_times[c_idx])
            {
                // next input is a measurement
                last_time = current_time ;
                current_time = control_times[c_idx] ;
                dt = current_time - last_time ;
                config.dt = dt ;
                setDeviceConfig(config) ;
                ZZ = allMeasurements[z_idx++] ;
                do_predict = true ;
            }
            else if ( measurement_times[z_idx] == control_times[c_idx])
            {
                // next input is odometry and measurement
                last_time = current_time ;
                current_time = control_times[c_idx] ;
                dt = current_time - last_time ;
                config.dt = dt ;
                setDeviceConfig(config) ;
                current_control = all_controls[c_idx++] ;
                ZZ = allMeasurements[z_idx++] ;
                do_predict = true ;
            }
            else
            {
                // next input is odometry

                // compute the time difference from last control
                last_time = current_time ;
                current_time = control_times[c_idx] ;
                dt = current_time - last_time ;
                config.dt = dt ;
                setDeviceConfig(config) ;
                current_control = all_controls[c_idx++] ;
                ZZ.clear();
                do_predict = true ;
            }
        }
        else
        {
            ZZ = allMeasurements[n] ;
            current_control = all_controls[n-1] ;
            dt = config.dt ;
            do_predict = true ;
        }

        if (config.followTrajectory){
            // load the next point in the trajectory
            for ( int i = 0 ; i < particles.n_particles ; i++ )
                particles.states[i] = trajVector[n] ;
        }
        else if ( n > 0 && do_predict )
        {
            // no motion for time step 1
            cout << "Performing vehicle prediction" << endl ;
            for ( int i = 0 ; i < config.subdividePredict ; i++ )
            {
                if ( config.motionType == CV_MOTION )
                    phdPredict(particles) ;
                else if ( config.motionType == ACKERMAN_MOTION )
                    phdPredict( particles, current_control ) ;
            }
        }
        if(config.savePrediction)
            writeParticlesMat(particles,n,"particles_predict");

        // need measurements from current time step for update
        if ( (ZZ.size() > 0) )
        {
            if (n==100)
            {
                cout << "serializing state at n = 100" << endl ;
                std::ofstream ofs("state100.bin") ;
                boost::archive::binary_oarchive oa(ofs) ;
                oa << particles ;
                oa << ZZ ;
            }
            cout << "Performing PHD Update" << endl ;
            particlesPreMerge = phdUpdateSynth(particles, ZZ) ;
        }
        cout << "Extracting SLAM state" << endl ;
        recoverSlamState(particles, expectedPose, cn_estimate ) ;

#ifdef DEBUG
        DEBUG_MSG( "Writing Log" ) ;
        writeParticlesMat(particles,n) ;
#endif

        nEff = 0 ;
        for ( int i = 0; i < particles.n_particles ; i++)
            nEff += exp(2*particles.weights[i]) ;
        nEff = 1.0/nEff/particles.n_particles ;
        DEBUG_VAL(nEff) ;
        if (nEff <= config.resampleThresh && ZZ.size() > 0 || particles.n_particles > 5*config.n_particles)
        {
            DEBUG_MSG("Resampling particles") ;
            particles = resampleParticles(particles,config.n_particles) ;
        }
        else
        {
            for ( int i = 0 ; i < particles.n_particles ; i++ )
            {
                particles.resample_idx[i] = i ;
            }
        }

        ZPrev = ZZ ;
        gettimeofday( &stop, NULL ) ;
        double elapsed = (stop.tv_sec - start.tv_sec)*1000 ;
        elapsed += (stop.tv_usec - start.tv_usec)/1000 ;
        fstream timeFile("loopTime.log", fstream::out|fstream::app ) ;
        timeFile << elapsed << endl ;
        timeFile.close() ;

        if ( isnan(nEff) )
        {
                cout << "nan weights detected! exiting..." << endl ;
                break ;
        }
    }
    }
    else
    {
        std::ifstream ifs("state100.bin") ;
        boost::archive::binary_iarchive ia(ifs) ;
        ia >> particles ;
        ia >> ZZ ;
        phdUpdateSynth(particles,ZZ) ;
    }
}

void run_disparity(){
    cout << "running with image data and disparity measurement model" << endl ;
    // load measurements
    vector<imageMeasurementSet> all_measurements ;
    string measurements_filename = data_dir + "measurements.txt" ;
    loadMeasurements(measurements_filename,all_measurements) ;
    if(n_steps < 0 )
        n_steps = all_measurements.size() ;

    // load trajectory, if required
    vector<ConstantVelocityState3D> trajVector ;
    if (config.followTrajectory){
        loadTrajectory(data_dir+"traj.txt",trajVector) ;
        // only need 1 particle
        config.n_particles = 1 ;
    }

    // recompute clutter density
    config.clutterDensity = config.clutterRate/
            (config.imageHeight*config.imageWidth) ;
    setDeviceConfig( config ) ;


    CameraState initial_state ;
    ConstantVelocityState3D expected_pose ;
    initial_state.pose.px = config.x0 ;
    initial_state.pose.py = config.y0 ;
    initial_state.pose.pz = config.z0 ;
    initial_state.pose.proll = config.roll0;
    initial_state.pose.ppitch = config.pitch0 ;
    initial_state.pose.pyaw  = config.yaw0 ;
    initial_state.pose.vx =  config.vx0 ;
    initial_state.pose.vy = config.vy0 ;
    initial_state.pose.vz = config.vz0 ;
    initial_state.pose.vroll = config.vroll0 ;
    initial_state.pose.vpitch = config.vpitch0;
    initial_state.pose.vyaw = config.vyaw0 ;
    initial_state.fx = config.fx ;
    initial_state.fy = config.fy ;
    initial_state.u0 = config.u0 ;
    initial_state.v0 = config.v0 ;
    DisparitySLAM slam(initial_state,config.n_particles) ;

    // initialize roll and yaw to be normally distributed, zero mean, ±5 deg
    for ( int n = 0 ; n < slam.n_particles ; n++ ){
        slam.states[n].pose.proll = initial_state.pose.proll + randn()*0.03 ;
        slam.states[n].pose.pyaw = initial_state.pose.ppitch + randn()*0.03 ;
    }

    for ( int k = 0 ; k < n_steps ; k++ ){
        // echo time
        gettimeofday( &start, NULL ) ;
        cout << "****** Time Step [" << k << "/" << n_steps << "] ******" << endl ;
        time( &rawtime ) ;
        timeinfo = localtime( &rawtime ) ;
        strftime(timestamp, 80, "%Y%m%d-%H%M%S", timeinfo ) ;
        cout << timestamp << endl ;

       // prediction
        if ( config.followTrajectory ){
            // load the next point in the trajectory
            for ( int n = 0 ; n < slam.n_particles ; n++ )
                slam.states[n].pose = trajVector[k] ;
        }
        else if ( k > 0 ){
            // no motion on first time step
            disparityPredict(slam) ;
        }

        // do measurement update
        disparityUpdate(slam,all_measurements[k]);


#ifdef DEBUG
        DEBUG_MSG( "Writing Log" ) ;
        writeParticlesMat(slam,k) ;
#endif

        // resample particles if nEff is below threshold
        double nEff = 0 ;
        for ( int i = 0; i < slam.n_particles ; i++)
            nEff += exp(2*slam.weights[i]) ;
        nEff = 1.0/nEff/slam.n_particles ;
        DEBUG_VAL(nEff) ;
        if (nEff <= config.resampleThresh )
        {
            DEBUG_MSG("Resampling particles") ;
            DEBUG_VAL(slam.n_particles) ;
            slam = resampleParticles(slam,config.n_particles) ;
            DEBUG_VAL(slam.n_particles) ;
        }
        else
        {
            for ( int i = 0 ; i < slam.n_particles ; i++ )
            {
                slam.resample_idx[i] = i ;
            }
        }


        if ( isnan(nEff) )
        {
            cout << "nan weights detected! exiting..." << endl ;
            break ;
        }

        cout << "Extracting SLAM state" << endl ;
        recoverSlamState(slam, expected_pose ) ;

        gettimeofday( &stop, NULL ) ;
        double elapsed = (stop.tv_sec - start.tv_sec)*1000 ;
        elapsed += (stop.tv_usec - start.tv_usec)/1000 ;
        fstream timeFile("loopTime.log", fstream::out|fstream::app ) ;
        timeFile << elapsed << endl ;
        timeFile.close() ;
    }
}

int main(int argc, char *argv[])
{
    // check cuda device properties
    int nDevices ;
    CUDA_SAFE_CALL(cudaGetDeviceCount( &nDevices )) ;
    cout << "Found " << nDevices << " CUDA Devices" << endl ;
    cudaDeviceProp props ;
    CUDA_SAFE_CALL(cudaGetDeviceProperties( &props, 0 )) ;
    cout << "Device name: " << props.name << endl ;
    cout << "Compute capability: " << props.major << "." << props.minor << endl ;
    deviceMemLimit = props.totalGlobalMem*0.95 ;
    cout << "Setting device memory limit to " << deviceMemLimit << " bytes" << endl ;

    // load the configuration file
    if ( argc < 2 )
    {
        cout << "missing configuration file argument" << endl ;
        exit(1) ;
    }
    DEBUG_MSG("Loading configuration file") ;
    config_filename = argv[1] ;
    loadConfig( config_filename.data() ) ;
    setDeviceConfig( config ) ;

    string run_type ;
    if ( argc >= 3 ){
        run_type = argv[2] ;
    }
    else{
        run_type = "synth" ;
    }

    bool profile_run = (argc > 3) ;
    if (profile_run){
        cout << "This is a profiling run" << endl ;
    }

#ifdef DEBUG
    // make a timestamped directory to store logfiles
    time( &rawtime ) ;
    timeinfo = localtime( &rawtime ) ;
    strftime(timestamp, 80, "%Y%m%d-%H%M%S", timeinfo ) ;
    mkdir(timestamp, S_IRWXU) ;
    // copy the configuration file into the log directory
    string copy_command("cp ") ;
    copy_command += config_filename ;
    copy_command += " " ;
    copy_command += timestamp ;
    system(copy_command.c_str()) ;
    string logdirname(timestamp) ;
#endif

    if (run_type.compare("synth")==0){
        run_synth(profile_run);
    }
    else if(run_type.compare("disparity")==0){
        run_disparity();
    }

#ifdef DEBUG
    // move all generated mat and log files to timestamped directory
    string command("mv *.mat ") ;
    command += logdirname ;
    system( command.c_str() ) ;
    command = "mv *.log " ;
    command += logdirname ;
    system( command.c_str() ) ;
#endif
    cout << "DONE!" << endl ;

    return 0 ;
}
