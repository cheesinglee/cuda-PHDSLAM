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
//#include "slamparams.h"
#include "slamtypes.h"
#include "phdfilter.h"
#include <cuda.h>
#include <cutil_inline.h>

#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/type_traits/conditional.hpp>

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

// measurement datafile
std::string measurementsFilename ;

// control input datafile
std::string controls_filename ;

// measurement and control timestamp datafiles
std::string measurements_time_filename ;
std::string controls_time_filename ;

char timestamp[80] ;

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

measurementSet parseMeasurements(string line)
{
    string value ;
    stringstream ss(line, ios_base::in) ;
    measurementSet v ;
    REAL range, bearing ;
    RangeBearingMeasurement *m ;
    while( ss.good() )
    {
        ss >> range ;
        ss >> bearing ;
        m = new RangeBearingMeasurement ;
        m->range = range ;
        m->bearing = bearing ;
        v.push_back(*m) ;
    }
    // TODO: sloppily remove the last invalid measurement (results from newline character?)
    v.pop_back() ;
    return v ;
}

std::vector<measurementSet> loadMeasurements( std::string filename )
{
    string line ;
    fstream measFile(filename.c_str()) ;
    std::vector<measurementSet> allMeasurements ;
    if (measFile.is_open())
    {
        // skip the header line
        getline(measFile,line) ;
        while(measFile.good())
        {
            getline(measFile,line) ;
            allMeasurements.push_back( parseMeasurements(line) ) ;
        }
        allMeasurements.pop_back() ;
        cout << "Loaded " << allMeasurements.size() << " measurements" << endl ;
    }
    else
        cout << "could not open measurements file!" << endl ;
    return allMeasurements ;
}

void printMeasurement(RangeBearingMeasurement z)
{
    cout << z.range <<"\t\t" << z.bearing << endl ;
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

void
writeParticlesMat(ParticleSLAM particles, int t = -1,
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
        write_map_mat( particles.maps_static, maps_static ) ;
        write_map_mat( particles.maps_dynamic, maps_dynamic ) ;

        // assemble final mat-file structure
//        DEBUG_MSG("assemble mat-file") ;
        const char* particleFieldNames[] = {"states","weights","maps_static",
                                            "maps_dynamic","resample_idx"} ;
//        DEBUG_MSG("mxCreateStructMatrix") ;
        mxArray* mxParticles = mxCreateStructMatrix(1,1,5,particleFieldNames) ;
//        DEBUG_MSG("states") ;
        mxSetFieldByNumber( mxParticles, 0, 0, states ) ;
//        DEBUG_MSG("weights") ;
        mxSetFieldByNumber( mxParticles, 0, 1, weights ) ;
//        DEBUG_MSG("maps_static") ;
        mxSetFieldByNumber( mxParticles, 0, 2, maps_static ) ;
//        DEBUG_MSG("maps_dynamic") ;
        mxSetFieldByNumber( mxParticles, 0, 3, maps_dynamic ) ;
//        DEBUG_MSG("resample_idx") ;
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

//void writeLogMat(ParticleSLAM<Gaussian4D> particles, ConstantVelocityState expectedPose,
//                                gaussianMixture expectedMap, vector<REAL> cn_estimate, int t)
//{
////        writeParticlesMat(particles,t) ;

////	fstream zFile("measurements.log", fstream::out|fstream::app ) ;
////	for ( unsigned int n = 0 ; n < Z.size() ; n++ )
////	{
////		zFile << Z[n].range << " " << Z[n].bearing << " ";
////	}
////	zFile << endl ;
////	zFile.close() ;
////

//        // create the filename
//        std::ostringstream oss ;
//                oss << "state_estimate" << t << ".mat" ;
//        std::string expectationFilename = oss.str() ;

//                const char* fieldNames[] = {"pose","map","cardinality","particles"} ;
//                mxArray* mxStates = mxCreateStructMatrix(1,1,4,fieldNames) ;

//        // pack data into mxArrays

//                // estimated pose
//        mxArray* mxPose = mxCreateNumericMatrix(6,1,mxDOUBLE_CLASS,mxREAL) ;
//        mxGetPr( mxPose )[0] = expectedPose.px ;
//        mxGetPr( mxPose )[1] = expectedPose.py ;
//        mxGetPr( mxPose )[2] = expectedPose.ptheta ;
//        mxGetPr( mxPose )[3] = expectedPose.vx ;
//        mxGetPr( mxPose )[4] = expectedPose.vy ;
//        mxGetPr( mxPose )[5] = expectedPose.vtheta ;

//                // estimated map
//        const char* mapFieldNames[] = {"weights","means","covs"} ;
//        mxArray* mxMap = mxCreateStructMatrix(1,1,3,mapFieldNames) ;
//        int nFeatures = expectedMap.size() ;
//        mwSize covDims[3] = {2,2,2} ;
//        covDims[2] = nFeatures ;
//        mxArray* mxWeights = mxCreateNumericMatrix(1,nFeatures,mxDOUBLE_CLASS,mxREAL) ;
//        mxArray* mxMeans = mxCreateNumericMatrix(2,nFeatures,mxDOUBLE_CLASS,mxREAL) ;
//        mxArray* mxCovs = mxCreateNumericArray(3,covDims,mxDOUBLE_CLASS,mxREAL) ;
//        if ( nFeatures > 0 )
//        {
//                for ( int i = 0 ; i < nFeatures ; i++ )
//                {
//                        mxGetPr( mxWeights )[i] = expectedMap[i].weight ;
//                        mxGetPr( mxMeans )[2*i+0] = expectedMap[i].mean[0] ;
//                        mxGetPr( mxMeans )[2*i+1] = expectedMap[i].mean[1] ;
//                        mxGetPr( mxCovs )[4*i+0] = expectedMap[i].cov[0] ;
//                        mxGetPr( mxCovs )[4*i+1] = expectedMap[i].cov[1] ;
//                        mxGetPr( mxCovs )[4*i+2] = expectedMap[i].cov[2] ;
//                        mxGetPr( mxCovs )[4*i+3] = expectedMap[i].cov[3] ;
//                }
//        }
//        mxSetFieldByNumber( mxMap, 0, 0, mxWeights ) ;
//        mxSetFieldByNumber( mxMap, 0, 1, mxMeans ) ;
//        mxSetFieldByNumber( mxMap, 0, 2, mxCovs ) ;

//                // estimated cardinality
//                mxArray* mx_cn = mxCreateNumericMatrix(1,config.maxCardinality+1,mxDOUBLE_CLASS,mxREAL) ;
//                for ( int n = 0 ; n <= config.maxCardinality ; n++ )
//                {
//                        mxGetPr( mx_cn )[n] = cn_estimate[n] ;
//                }

//                // particle poses and weights
//                int n_particles = particles.nParticles ;
//                const char* particle_field_names[] = {"weights","poses"} ;
//                mxArray* mx_particles = mxCreateStructMatrix(1,1,2,particle_field_names) ;
//                mxArray* mx_particle_weights = mxCreateNumericMatrix(1,n_particles,mxDOUBLE_CLASS,mxREAL) ;
//                mxArray* mx_particle_poses = mxCreateNumericMatrix(6,n_particles,mxDOUBLE_CLASS,mxREAL) ;
//                for ( int i = 0 ; i < n_particles ; i++ )
//                {
//                        mxGetPr( mx_particle_weights )[i] = particles.weights[i] ;
//                        mxGetPr( mx_particle_poses )[6*i+0] = particles.states[i].px ;
//                        mxGetPr( mx_particle_poses )[6*i+1] = particles.states[i].py ;
//                        mxGetPr( mx_particle_poses )[6*i+2] = particles.states[i].ptheta ;
//                        mxGetPr( mx_particle_poses )[6*i+3] = particles.states[i].vx ;
//                        mxGetPr( mx_particle_poses )[6*i+4] = particles.states[i].vy ;
//                        mxGetPr( mx_particle_poses )[6*i+5] = particles.states[i].vtheta ;
//                }
//                mxSetFieldByNumber( mx_particles,0,0,mx_particle_weights ) ;
//                mxSetFieldByNumber( mx_particles,0,1,mx_particle_poses) ;
////	// resize the array to accommodate the new entry
////	mxSetM( mxStates, t+1 ) ;

//        // save the new entry
//        mxSetFieldByNumber( mxStates, 0, 0, mxPose ) ;
//        mxSetFieldByNumber( mxStates, 0, 1, mxMap ) ;
//                mxSetFieldByNumber( mxStates, 0, 2, mx_cn) ;
//                mxSetFieldByNumber( mxStates, 0, 3, mx_particles) ;

//        // write to the mat-file
//        MATFile* expectationFile = matOpen( expectationFilename.c_str(), "w") ;
//        matPutVariable( expectationFile, "expectation", mxStates ) ;
//        matClose( expectationFile ) ;

//        // clean up
//        mxDestroyArray( mxStates ) ;
//}

//void writeParticles(ParticleSLAM particles, std::string filename, int t = -1)
//{
//        std::ostringstream oss ;
//        oss << filename ;
//        oss << setfill('0') << setw(5) ;
//        oss << t << ".log" ;
//        std::string particlesFilename = oss.str() ;
////        if ( t >= 0 )
////        {
////                char timeStep[8] ;
////                sprintf(timeStep,"%d",t) ;
////                particlesFilename += timeStep ;
////        }
////        particlesFilename += ".log" ;
//        fstream particlesFile(particlesFilename.c_str(), fstream::out|fstream::app ) ;
//        if (!particlesFile)
//        {
//                cout << "failed to open log file" << endl ;
//                return ;
//        }
//        for ( int n = 0 ; n < particles.nParticles ; n++ )
//        {
//                particlesFile << particles.weights[n] << " "
//                                << particles.states[n].px << " "
//                                << particles.states[n].py << " "
//                                << particles.states[n].ptheta << " "
//                                << particles.states[n].vx << " "
//                                << particles.states[n].vy << " "
//                                << particles.states[n].vtheta << " " ;
//                for ( int i = 0 ; i < (int)particles.maps[n].size() ; i++ )
//                {
//                        particlesFile << particles.maps[n][i].weight << " "
//                                        << particles.maps[n][i].mean[0] << " "
//                                        << particles.maps[n][i].mean[1] << " "
//                                        << particles.maps[n][i].cov[0] << " "
//                                        << particles.maps[n][i].cov[1] << " "
//                                        << particles.maps[n][i].cov[2] << " "
//                                        << particles.maps[n][i].cov[3] << " " ;
//                }
//                particlesFile << endl ;
//        }
//        particlesFile.close() ;
//}

//ParticleSLAM loadParticles(std::string filename)
//{
//    ifstream file( filename.c_str() ) ;
//    int n_particles = std::count(std::istreambuf_iterator<char>(file),
//                                std::istreambuf_iterator<char>(), '\n');
//    ParticleSLA particles(n_particles) ;
//    cout << "reading " << n_particles << " particles from " << filename << endl ;
//    file.seekg(ifstream::beg) ;
//    for ( int n = 0 ; n < n_particles ; n++ )
//    {
//        std::string line ;
//        getline(file,line) ;
////        cout << line << endl ;
//        istringstream iss(line) ;
//        iss >> particles.weights[n] >> particles.states[n].px
//            >> particles.states[n].py >> particles.states[n].ptheta
//            >> particles.states[n].vx >> particles.states[n].vy
//            >> particles.states[n].vtheta ;
//        while ( !iss.eof() )
//        {
//            Gaussian4D feature ;
//            iss >> feature.weight >> feature.mean[0] >> feature.mean[1]
//                >> feature.cov[0] >> feature.cov[1] >> feature.cov[2]
//                >> feature.cov[3] ;
//            particles.maps[n].push_back(feature) ;
//        }
//        particles.maps[n].pop_back() ;
////        cout << "read " << particles.maps[n].size() << " features" << endl ;
//    }

//    return particles ;
//}

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

void writeLog(const ParticleSLAM& particles, ConstantVelocityState expectedPose,
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

//void writeParticlesPickle( ParticleSLAM<Gaussian4D> particles, const char* filename, int t = -1 )
//{
//	// create the filename
//	std::string particlesFilename(filename) ;
//	if ( t >= 0 )
//	{
//			char timeStep[8] ;
//			sprintf(timeStep,"%d",t) ;
//			particlesFilename += timeStep ;
//	}
//	particlesFilename += ".p2" ;

//	Arr weights_arr(particles.nParticles) ;
//	Arr poses_arr(particles.nParticles) ;
//	Arr maps_arr( particles.nParticles ) ;
//	for ( int i = 0 ; i < particles.nParticles ; i++ )
//	{
//		weights_arr.append( particles.weights[i] ) ;

//		Arr single_pose(6) ;
//		single_pose.append( particles.states[i].px ) ;
//		single_pose.append( particles.states[i].py ) ;
//		single_pose.append( particles.states[i].ptheta );
//		single_pose.append( particles.states[i].vx ) ;
//		single_pose.append( particles.states[i].vy ) ;
//		single_pose.append( particles.states[i].vtheta );
//		poses_arr.append( single_pose ) ;

//		Tab single_map ;
//		int map_size = particles.maps[i].size() ;
//		Arr map_weights_arr(map_size) ;
//		Arr map_means_arr(map_size*2) ;
//		Arr map_covs_arr(map_size*4) ;
//		for ( int j = 0 ; j < map_size ; j++ )
//		{
//			map_weights_arr.append( particles.maps[i][j].weight ) ;
//			map_means_arr.append( particles.maps[i][j].mean[0] ) ;
//			map_means_arr.append( particles.maps[i][j].mean[1] ) ;
//			map_covs_arr.append( particles.maps[i][j].cov[0] ) ;
//			map_covs_arr.append( particles.maps[i][j].cov[1] ) ;
//			map_covs_arr.append( particles.maps[i][j].cov[2] ) ;
//			map_covs_arr.append( particles.maps[i][j].cov[3] ) ;
//		}
//		single_map["weights"] = map_weights_arr ;
//		single_map["means"] = map_means_arr ;
//		single_map["covs"] = map_covs_arr ;
//		maps_arr.append( single_map ) ;
//	}
//	Tab particles_tab ;
//	particles_tab["weights"] = weights_arr ;
//	particles_tab["poses"] = poses_arr ;
//	particles_tab["maps"] = maps_arr ;

//	DumpValToFile( particles_tab, particlesFilename, SERIALIZE_P2 ) ;
//}

void loadConfig(const char* filename)
{
    using namespace boost::program_options ;
    options_description desc("SLAM filter config") ;
    desc.add_options()
            ("initial_x", value<REAL>(&config.x0)->default_value(30), "Initial x position")
            ("initial_y", value<REAL>(&config.y0)->default_value(5), "Initial y position")
            ("initial_theta", value<REAL>(&config.theta0)->default_value(0), "Initial heading")
            ("initial_vx", value<REAL>(&config.vx0)->default_value(7), "Initial x velocity")
            ("initial_vy", value<REAL>(&config.vy0)->default_value(0), "Initial y velocity")
            ("initial_vtheta", value<REAL>(&config.vtheta0)->default_value(0.3142), "Initial heading velocity")
            ("motion_type", value<int>(&config.motionType)->default_value(1), "0 = Constant Velocity, 1 = Ackerman steering")
            ("acc_x", value<REAL>(&config.ax)->default_value(0.5), "Standard deviation of x acceleration")
            ("acc_y", value<REAL>(&config.ay)->default_value(0), "Standard deviation of y acceleration")
            ("acc_theta", value<REAL>(&config.atheta)->default_value(0.0087), "Standard deviation of theta acceleration")
            ("dt", value<REAL>(&config.dt)->default_value(0.1), "Duration of each timestep")
            ("max_bearing", value<REAL>(&config.maxBearing)->default_value(M_PI), "Maximum sensor bearing")
            ("min_range", value<REAL>(&config.minRange)->default_value(0), "Minimum sensor range")
            ("max_range", value<REAL>(&config.maxRange)->default_value(20), "Maximum sensor range")
            ("std_bearing", value<REAL>(&config.stdBearing)->default_value(0.0524), "Standard deviation of sensor bearing noise")
            ("std_range", value<REAL>(&config.stdRange)->default_value(1.0), "Standard deviation of sensor range noise")
            ("clutter_rate", value<REAL>(&config.clutterRate)->default_value(15), "Poisson mean number of clutter measurements per scan")
            ("pd", value<REAL>(&config.pd)->default_value(0.98), "Nominal probability of detection for in-range features")
            ("ps", value<REAL>(&config.ps)->default_value(0.8), "Nominal probability of survival for dynamic features")
            ("n_particles", value<int>(&config.n_particles)->default_value(512), "Number of vehicle pose particles")
			("n_predict_particles", value<int>(&config.nPredictParticles)->default_value(1), "Number of new vehicle pose particles to spawn for each prior particle when doing prediction")
            ("resample_threshold", value<REAL>(&config.resampleThresh)->default_value(0.15), "Threshold on normalized nEff for particle resampling")
            ("subdivide_predict", value<int>(&config.subdividePredict)->default_value(1), "Perform the prediction over several shorter time intervals before the update")
            ("birth_weight", value<REAL>(&config.birthWeight)->default_value(0.05), "Weight of birth features")
			("birth_noise_factor", value<REAL>(&config.birthNoiseFactor)->default_value(1.5), "Factor which multiplies the measurement noise to determine covariance of birth features")
            ("gate_births", value<bool>(&config.gateBirths)->default_value(true), "Enable measurement gating on births")
            ("gate_measurements", value<bool>(&config.gateMeasurements)->default_value(true), "Gate measurements for update")
            ("gate_threshold", value<REAL>(&config.gateThreshold)->default_value(10), "Mahalanobis distance threshold for gating")
            ("dynamic_features", value<bool>(&config.dynamicFeatures)->default_value(false), "Use dynamic model for map features")
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
            ("tau", value<REAL>(&config.tau)->default_value(0), "Velocity threshold for jump markov transition probability")
            ("beta", value<REAL>(&config.beta)->default_value(1), "Steepness of sigmoid function for computing JMM transition probability")
            ("measurements_filename", value<std::string>(&measurementsFilename)->default_value("measurements.txt"), "Path to measurements datafile")
            ("controls_filename", value<std::string>(&controls_filename)->default_value("controls.txt"), "Path to controls datafile")
            ("measurements_time_filename", value<std::string>(&measurements_time_filename)->default_value(""), "Path to measurement timestamps datafile")
            ("controls_time_filename", value<std::string>(&controls_time_filename)->default_value(""), "Path to control timestamps datafile")
            ("max_time_steps", value<int>(&config.maxSteps)->default_value(10000), "Limit the number of time steps to execute")
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

//            variables_map::iterator it = vm.begin() ;
//            while ( it != vm.end() )
//            {
//                cout << it->first << " = " << boost::lexical_cast<double>(it->second.value()) << endl ;
//                it++ ;
//            }

        }
        catch( std::exception& e )
        {
            cout << "Error parsing config file: " << e.what() << endl ;
//            exit(1);
        }
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

    // time variables
    time_t rawtime ;
    struct tm *timeinfo ;

    // load measurement data
    std::vector<measurementSet> allMeasurements ;
    allMeasurements = loadMeasurements(measurementsFilename) ;
    std::vector<measurementSet>::iterator i( allMeasurements.begin() ) ;
    std::vector<RangeBearingMeasurement>::iterator ii ;

    // load control inputs
    vector<AckermanControl> all_controls ;
    all_controls = loadControls( controls_filename ) ;

    // load timestamps
    vector<REAL> measurement_times = loadTimestamps( measurements_time_filename ) ;
    vector<REAL> control_times = loadTimestamps( controls_time_filename ) ;
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

        nSteps = measurement_times.size() + control_times.size() ;
    }
    if (nSteps > config.maxSteps && config.maxSteps > 0)
        nSteps = config.maxSteps ;

    // initialize particles
    ParticleSLAM particles(config.n_particles) ;
    for (int n = 0 ; n < config.n_particles ; n++ )
    {
        particles.states[n].px = config.x0 ;
        particles.states[n].py = config.y0 ;
        particles.states[n].ptheta = config.theta0 ;
        particles.states[n].vx = config.vx0 ;
        particles.states[n].vy = config.vy0 ;
        particles.states[n].vtheta = config.vtheta0 ;
        particles.weights[n] = -log(config.n_particles) ;
        if ( config.filterType == CPHD_TYPE )
        {
            particles.cardinalities[n].assign( config.maxCardinality+1, -log(config.maxCardinality+1) ) ;
        }
    }

    if ( config.filterType == CPHD_TYPE )
    {
            particles.cardinality_birth.assign( config.maxCardinality+1, LOG0 ) ;
            particles.cardinality_birth[0] = 0 ;
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
#endif

    // do the simulation
    measurementSet ZZ ;
    measurementSet ZPrev ;
    ParticleSLAM particlesPreMerge(particles) ;
    ConstantVelocityState expectedPose ;
    vector<Gaussian2D> expected_map_static ;
    vector<Gaussian4D> expected_map_dynamic ;
    vector<REAL> cn_estimate ;
    REAL nEff ;
    timeval start, stop ;
    REAL current_time = 0 ;
    REAL last_time = 0 ;
    REAL dt = 0 ;
    int z_idx = 0 ;
    int c_idx = 0 ;
    AckermanControl current_control ;
    current_control.alpha = 0 ;
    current_control.v_encoder = 0 ;
    bool do_predict = false ;
    string logdirname(timestamp) ;

    cout << "STARTING SIMULATION" << endl ;
    if ( config.filterType == CPHD_TYPE )
    {
        DEBUG_MSG("Initializing CPHD constants") ;
        initCphdConstants() ;
    }

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
            if (measurement_times[z_idx] < control_times[c_idx])
            {
                // next input is a measurement
                ZZ = allMeasurements[z_idx++] ;
                do_predict = false ;
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

        // no motion for time step 1
        if ( n > 0 && do_predict )
        {
            cout << "Performing vehicle prediction" << endl ;
            for ( int i = 0 ; i < config.subdividePredict ; i++ )
            {
                if ( config.motionType == CV_MOTION )
                    phdPredict(particles) ;
                else if ( config.motionType == ACKERMAN_MOTION )
                    phdPredict( particles, current_control ) ;
            }
        }

        // need measurements from current time step for update
        if ( (ZZ.size() > 0) )
        {
            cout << "Performing PHD Update" << endl ;
            particlesPreMerge = phdUpdate(particles, ZZ) ;
        }
        cout << "Extracting SLAM state" << endl ;
        recoverSlamState(particles, expectedPose, expected_map_static,
                         expected_map_dynamic, cn_estimate ) ;

#ifdef DEBUG
        DEBUG_MSG( "Writing Log" ) ;
        writeParticlesMat(particles,n) ;
//        writeLog(particles, expectedPose, expected_map_static,
//                 expected_map_dynamic,idx_resample, cn_estimate, n) ;
//                writeParticles(particles,"particles",n) ;
#endif

        nEff = 0 ;
        for ( int i = 0; i < particles.n_particles ; i++)
            nEff += exp(2*particles.weights[i]) ;
        nEff = 1.0/nEff/particles.n_particles ;
        DEBUG_VAL(nEff) ;
        if (nEff <= config.resampleThresh && ZZ.size() > 0 )
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
#ifdef DEBUG
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