#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
#include <sstream>

#include <mex.h>
#include <mat.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/time.h>
#include "slamparams.h"
#include "slamtypes.h"
#include <cuda.h>
#include <cutil_inline.h>

#include <boost/program_options.hpp>

#define DEBUG

#ifdef DEBUG
    #define DEBUG_MSG(x) cout <<"["<< __func__ << "]: " << x << endl
    #define DEBUG_VAL(x) cout << "["<<__func__ << "]: " << #x << " = " << x << endl ;
#else
    #define DEBUG_MSG(x)
    #define DEBUG_VAL(x)
#endif

using namespace std ;

//--- Externally defined CUDA kernel callers
extern "C"
void
phdPredict( ParticleSLAM& particles ) ;

extern "C"
void
addBirths( ParticleSLAM& particles, measurementSet ZPrev ) ;

extern "C"
ParticleSLAM
phdUpdate(ParticleSLAM& particles, measurementSet measurements) ;

extern "C"
ParticleSLAM resampleParticles( ParticleSLAM oldParticles ) ;

extern "C"
void recoverSlamState(ParticleSLAM particles, ConstantVelocityState *expectedPose,
        gaussianMixture *expectedMap) ;

extern "C"
void setDeviceConfig( const SlamConfig& config ) ;
//--- End external functions

// SLAM configuration
SlamConfig config ;

// device memory limit
size_t deviceMemLimit ;

// measurement datafile
std::string measurementsFilename ;

char timestamp[80] ;

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
    }
    else
        cout << "could not open measurements file!" << endl ;
    return allMeasurements ;
}

void printMeasurement(RangeBearingMeasurement z)
{
    cout << z.range <<"\t\t" << z.bearing << endl ;
}

void
writeParticlesMat(ParticleSLAM particles, int t = -1, const char* filename="particles")
{
        // create the filename
        std::string particlesFilename(filename) ;
        if ( t >= 0 )
        {
                char timeStep[8] ;
                sprintf(timeStep,"%d",t) ;
                particlesFilename += timeStep ;
        }
        particlesFilename += ".mat" ;

        // load particles into mxArray object
        mwSize nParticles = particles.nParticles ;

        mxArray* states = mxCreateNumericMatrix(6,nParticles,mxDOUBLE_CLASS,mxREAL) ;
        double* statesArray = (double*)mxCalloc(nParticles*6,sizeof(double));
        int i = 0 ;
        for ( unsigned int p = 0 ; p < nParticles ; p++ )
        {
                statesArray[i+0] = particles.states[p].px ;
                statesArray[i+1] = particles.states[p].py ;
                statesArray[i+2] = particles.states[p].ptheta ;
                statesArray[i+3] = particles.states[p].vx ;
                statesArray[i+4] = particles.states[p].vy ;
                statesArray[i+5] = particles.states[p].vtheta ;
                i+=6 ;
        }
        mxFree(mxGetPr(states)) ;
        mxSetPr(states,statesArray) ;

        mxArray* weights = mxCreateNumericMatrix(nParticles,1,mxDOUBLE_CLASS,mxREAL) ;
        double* weightsArray = (double*)mxCalloc(nParticles,sizeof(double)) ;
        std::copy( particles.weights.begin(), particles.weights.end(), weightsArray ) ;
//        memcpy(weightsArray,particles.weights,nParticles*sizeof(double)) ;
        mxFree(mxGetPr(weights)) ;
        mxSetPr(weights,weightsArray) ;

        const char* mapFieldNames[] = {"weights","means","covs"} ;
        mxArray* maps = mxCreateStructMatrix(nParticles,1,3,mapFieldNames) ;
        mwSize covDims[3] = {2,2,2} ;
        mxArray* mapWeights ;
        mxArray* mapMeans ;
        mxArray* mapCovs ;
        for ( unsigned int p = 0 ; p < nParticles ; p++ )
        {
                gaussianMixture map = particles.maps[p] ;
                mwSize mapSize = map.size() ;
                covDims[2] = mapSize ;
                mapWeights = mxCreateNumericMatrix(1,mapSize,mxDOUBLE_CLASS,mxREAL) ;
                mapMeans = mxCreateNumericMatrix(2,mapSize,mxDOUBLE_CLASS,mxREAL) ;
                mapCovs = mxCreateNumericArray(3,covDims,mxDOUBLE_CLASS,mxREAL) ;
                if ( mapSize > 0 )
                {
                        for ( unsigned int j = 0 ; j < mapSize ; j++ )
                        {
                                mxGetPr( mapWeights )[j] = map[j].weight ;
                                mxGetPr( mapMeans )[2*j+0] = map[j].mean[0] ;
                                mxGetPr( mapMeans )[2*j+1] = map[j].mean[1] ;
                                mxGetPr( mapCovs )[4*j+0] = map[j].cov[0] ;
                                mxGetPr( mapCovs )[4*j+1] = map[j].cov[1] ;
                                mxGetPr( mapCovs )[4*j+2] = map[j].cov[2] ;
                                mxGetPr( mapCovs )[4*j+3] = map[j].cov[3] ;
                        }
                }
                mxSetFieldByNumber( maps, p, 0, mapWeights ) ;
                mxSetFieldByNumber( maps, p, 1, mapMeans ) ;
                mxSetFieldByNumber( maps, p, 2, mapCovs ) ;
        }

        const char* particleFieldNames[] = {"states","weights","maps"} ;
        mxArray* mxParticles = mxCreateStructMatrix(1,1,3,particleFieldNames) ;
        mxSetFieldByNumber( mxParticles, 0, 0, states ) ;
        mxSetFieldByNumber( mxParticles, 0, 1, weights ) ;
        mxSetFieldByNumber( mxParticles, 0, 2, maps ) ;

        // write to mat file
        MATFile* matfile = matOpen( particlesFilename.c_str(), "w" ) ;
        matPutVariable( matfile, "particles", mxParticles ) ;
        matClose(matfile) ;

        // clean up
        mxDestroyArray( mxParticles ) ;
}

void writeLogMat(ParticleSLAM particles,
                measurementSet Z, ConstantVelocityState expectedPose,
                gaussianMixture expectedMap, int t)
{
        writeParticlesMat(particles,t) ;

//	fstream zFile("measurements.log", fstream::out|fstream::app ) ;
//	for ( unsigned int n = 0 ; n < Z.size() ; n++ )
//	{
//		zFile << Z[n].range << " " << Z[n].bearing << " ";
//	}
//	zFile << endl ;
//	zFile.close() ;
//

        // create the filename
        std::ostringstream oss ;
        oss << "expectation" << t << ".mat" ;
        std::string expectationFilename = oss.str() ;

        const char* fieldNames[] = {"pose","map"} ;
        mxArray* mxStates = mxCreateStructMatrix(1,1,2,fieldNames) ;

//	// create the expected state mat file if it doesn't exist
//	MATFile* expectationFile ;
//	if ( !(expectationFile = matOpen("expectation.mat", "u" )) )
//		expectationFile = matOpen("expectation.mat", "w" ) ;
//
//	// create the structure of expected states if it doesn't exist
//	mxArray* mxStates ;
//	if ( !(mxStates = matGetVariable( expectationFile, "expectation" ) ) )
//	{
//		const char* fieldNames[] = {"pose","map"} ;
//		mxStates = mxCreateStructMatrix(1,1,2,fieldNames) ;
//	}

        // pack data into mxArrays
        mxArray* mxPose = mxCreateNumericMatrix(6,1,mxDOUBLE_CLASS,mxREAL) ;
        mxGetPr( mxPose )[0] = expectedPose.px ;
        mxGetPr( mxPose )[1] = expectedPose.py ;
        mxGetPr( mxPose )[2] = expectedPose.ptheta ;
        mxGetPr( mxPose )[3] = expectedPose.vx ;
        mxGetPr( mxPose )[4] = expectedPose.vy ;
        mxGetPr( mxPose )[5] = expectedPose.vtheta ;

        const char* mapFieldNames[] = {"weights","means","covs"} ;
        mxArray* mxMap = mxCreateStructMatrix(1,1,3,mapFieldNames) ;
        int nFeatures = expectedMap.size() ;
        mwSize covDims[3] = {2,2,2} ;
        covDims[2] = nFeatures ;
        mxArray* mxWeights = mxCreateNumericMatrix(1,nFeatures,mxDOUBLE_CLASS,mxREAL) ;
        mxArray* mxMeans = mxCreateNumericMatrix(2,nFeatures,mxDOUBLE_CLASS,mxREAL) ;
        mxArray* mxCovs = mxCreateNumericArray(3,covDims,mxDOUBLE_CLASS,mxREAL) ;
        if ( nFeatures > 0 )
        {
                for ( int i = 0 ; i < nFeatures ; i++ )
                {
                        mxGetPr( mxWeights )[i] = expectedMap[i].weight ;
                        mxGetPr( mxMeans )[2*i+0] = expectedMap[i].mean[0] ;
                        mxGetPr( mxMeans )[2*i+1] = expectedMap[i].mean[1] ;
                        mxGetPr( mxCovs )[4*i+0] = expectedMap[i].cov[0] ;
                        mxGetPr( mxCovs )[4*i+1] = expectedMap[i].cov[1] ;
                        mxGetPr( mxCovs )[4*i+2] = expectedMap[i].cov[2] ;
                        mxGetPr( mxCovs )[4*i+3] = expectedMap[i].cov[3] ;
                }
        }

        mxSetFieldByNumber( mxMap, 0, 0, mxWeights ) ;
        mxSetFieldByNumber( mxMap, 0, 1, mxMeans ) ;
        mxSetFieldByNumber( mxMap, 0, 2, mxCovs ) ;

//	// resize the array to accommodate the new entry
//	mxSetM( mxStates, t+1 ) ;

        // save the new entry
        mxSetFieldByNumber( mxStates, 0, 0, mxPose ) ;
        mxSetFieldByNumber( mxStates, 0, 1, mxMap ) ;

        // write to the mat-file
        MATFile* expectationFile = matOpen( expectationFilename.c_str(), "w") ;
        matPutVariable( expectationFile, "expectation", mxStates ) ;
        matClose( expectationFile ) ;

        // clean up
        mxDestroyArray( mxStates ) ;
}

void writeParticles(ParticleSLAM particles, const char* filename, int t = -1)
{
        std::string particlesFilename(filename) ;
        if ( t >= 0 )
        {
                char timeStep[8] ;
                sprintf(timeStep,"%d",t) ;
                particlesFilename += timeStep ;
        }
        particlesFilename += ".log" ;
        fstream particlesFile(particlesFilename.c_str(), fstream::out|fstream::app ) ;
        if (!particlesFile)
        {
                cout << "failed to open log file" << endl ;
                return ;
        }
        for ( int n = 0 ; n < particles.nParticles ; n++ )
        {
                particlesFile << particles.weights[n] << " "
                                << particles.states[n].px << " "
                                << particles.states[n].py << " "
                                << particles.states[n].ptheta << " "
                                << particles.states[n].vx << " "
                                << particles.states[n].vy << " "
                                << particles.states[n].vtheta << " " ;
                for ( int i = 0 ; i < (int)particles.maps[n].size() ; i++ )
                {
                        particlesFile << particles.maps[n][i].weight << " "
                                        << particles.maps[n][i].mean[0] << " "
                                        << particles.maps[n][i].mean[1] << " "
                                        << particles.maps[n][i].cov[0] << " "
                                        << particles.maps[n][i].cov[1] << " "
                                        << particles.maps[n][i].cov[2] << " "
                                        << particles.maps[n][i].cov[3] << " " ;
                }
                particlesFile << endl ;
        }
        particlesFile.close() ;
}

void writeLog(ParticleSLAM particles,
                measurementSet Z, ConstantVelocityState expectedPose,
                gaussianMixture expectedMap, int t)
{
        writeParticles(particles,"particles",t) ;

        fstream zFile("measurements.log", fstream::out|fstream::app ) ;
        for ( unsigned int n = 0 ; n < Z.size() ; n++ )
        {
                zFile << Z[n].range << " " << Z[n].bearing << " ";
        }
        zFile << endl ;
        zFile.close() ;

        fstream stateFile("expectation.log", fstream::out|fstream::app ) ;
        stateFile << expectedPose.px << " " << expectedPose.py << " "
                        << expectedPose.ptheta << " " << expectedPose.vx << " "
                        << expectedPose.vy << " " << expectedPose.vtheta << " " ;
        for ( int n = 0 ; n < (int)expectedMap.size() ; n++ )
        {
                stateFile << expectedMap[n].weight << " "
                                << expectedMap[n].mean[0] << " "
                                << expectedMap[n].mean[1] << " "
                                << expectedMap[n].cov[0] << " "
                                << expectedMap[n].cov[1] << " "
                                << expectedMap[n].cov[2] << " "
                                << expectedMap[n].cov[3] << " " ;
        }
        stateFile << endl ;
        stateFile.close() ;
}

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
            ("acc_x", value<REAL>(&config.ax)->default_value(0.5), "Standard deviation of x acceleration")
            ("acc_y", value<REAL>(&config.ay)->default_value(0), "Standard deviation of y acceleration")
            ("acc_theta", value<REAL>(&config.atheta)->default_value(0.0087), "Standard deviation of theta acceleration")
            ("n_alpha", value<REAL>(&config.n_alpha)->default_value(0.087), "Standard deviation of steering wheel angle noise (Ackerman steering)")
            ("n_encoder", value<REAL>(&config.n_encoder)->default_value(2), "Standard deviation of wheel encoder noise (Ackerman steering)")
            ("dt", value<REAL>(&config.dt)->default_value(0.1), "Duration of each timestep")
            ("max_bearing", value<REAL>(&config.maxBearing)->default_value(M_PI), "Maximum sensor bearing")
            ("max_range", value<REAL>(&config.maxRange)->default_value(20), "Maximum sensor range")
            ("std_bearing", value<REAL>(&config.stdBearing)->default_value(0.0524), "Standard deviation of sensor bearing noise")
            ("std_range", value<REAL>(&config.stdRange)->default_value(1.0), "Standard deviation of sensor range noise")
            ("clutter_rate", value<REAL>(&config.clutterRate)->default_value(15), "Poisson mean number of clutter measurements per scan")
            ("pd", value<REAL>(&config.pd)->default_value(0.98), "Nominal probability of detection for in-range features")
            ("n_particles", value<int>(&config.nParticles)->default_value(512), "Number of vehicle pose particles")
            ("resample_threshold", value<REAL>(&config.resampleThresh)->default_value(0.15), "Threshold on normalized nEff for particle resampling")
            ("birth_weight", value<REAL>(&config.birthWeight)->default_value(0.05), "Weight of birth features")
            ("gated_births", value<bool>(&config.gatedBirths)->default_value(true), "Enable measurement gating on births")
            ("min_expected_feature_weight", value<REAL>(&config.minExpectedFeatureWeight)->default_value(0.33), "Minimum feature weight for expected map")
            ("min_separation", value<REAL>(&config.minSeparation)->default_value(5), "Minimum Mahalanobis separation between features")
            ("max_features", value<int>(&config.maxFeatures)->default_value(100), "Maximum number of features in map")
            ("min_feature_weight", value<REAL>(&config.minFeatureWeight)->default_value(0.00001), "Minimum feature weight")
            ("particle_weighting", value<int>(&config.particleWeighting)->default_value(1), "Particle weighting scheme: 1 = cluster process 2 = Vo's")
            ("measurements_filename", value<std::string>(&measurementsFilename)->default_value("measurements.txt"), "Path to measurements datafile")
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
            vm.notify() ;
            // compute clutter density
            config.clutterDensity = config.clutterRate/
                                    ( pow(config.maxRange,2)*config.maxBearing ) ;
            #ifdef DEBUG
            variables_map::iterator it ;
            for ( it = vm.begin() ; it != vm.end() ; it++ )
            {
                cout << it->first << " => " << it->second.as<REAL>() << endl ;
            }
            #endif
        }
        catch( std::exception& e )
        {
            cout << "Error parsing config file: " << e.what() << endl ;
//            exit(1);
        }
    }

//    fstream cfgFile(filename) ;
//    string line ;
//    string key ;
//    REAL val ;
//    int eqIdx ;
//    while( cfgFile.good() )
//    {
//        getline( cfgFile, line ) ;
//        eqIdx = line.find("=") ;
//        if ( eqIdx != string::npos )
//        {
//            line.replace(eqIdx,1," ") ;
//            istringstream iss(line) ;
//            iss >> key >> val ;
//            config.insert( pair<string,REAL>(key,val) ) ;
//        }
//    }
//    cfgFile.close() ;
//#ifdef DEBUG
//    filterConfig::iterator it ;
//    for ( it = config.begin() ; it != config.end() ; it++ )
//    {
//        cout << it->first << " = " << it->second << endl ;
//    }
//#endif
}

int main(int argc, char *argv[])
{
#ifdef DEBUG
        time_t rawtime ;
        struct tm *timeinfo ;
        time( &rawtime ) ;
        timeinfo = localtime( &rawtime ) ;
        strftime(timestamp, 80, "%Y%m%d-%H%M%S", timeinfo ) ;
        mkdir(timestamp, S_IRWXU) ;
#endif

        // load the configuration file
        if ( argc < 2 )
        {
            cout << "missing configuration file argument" << endl ;
            exit(1) ;
        }
        DEBUG_MSG("Loading configuration file") ;
        loadConfig( argv[1] ) ;
        setDeviceConfig( config ) ;

        // load measurement data
        std::vector<measurementSet> allMeasurements ;
        allMeasurements = loadMeasurements(measurementsFilename) ;
        std::vector<measurementSet>::iterator i( allMeasurements.begin() ) ;
        std::vector<RangeBearingMeasurement>::iterator ii ;
        int nSteps = allMeasurements.size() ;

//        // initialize the random number generator
//        if ( !init_sprng(DEFAULT_RNG_TYPE, make_sprng_seed(),SPRNG_DEFAULT ) )
//                cout << "Error initializing SPRNG" << endl ;

        // initialize particles
        ParticleSLAM particles( config.nParticles ) ;
        for (int n = 0 ; n < config.nParticles ; n++ )
        {
                particles.states[n].px = INITPX ;
                particles.states[n].py = INITPY ;
                particles.states[n].ptheta = INITPTHETA ;
                particles.states[n].vx = INITVX ;
                particles.states[n].vy = INITVY ;
                particles.states[n].vtheta = INITVTHETA ;
                particles.weights[n] = 1.0/config.nParticles ;
        }

        // check cuda device properties
        int nDevices ;
        cudaGetDeviceCount( &nDevices ) ;
        cout << "Found " << nDevices << " CUDA Devices" << endl ;
        cudaDeviceProp props ;
        cudaGetDeviceProperties( &props, 0 ) ;
        cout << "Device name: " << props.name << endl ;
        cout << "Compute capability: " << props.major << "." << props.minor << endl ;
        deviceMemLimit = props.totalGlobalMem*0.95 ;
        cout << "Setting device memory limit to " << deviceMemLimit << " bytes" << endl ;

//	cudaPrintfInit() ;

        // do the simulation
        measurementSet ZZ ;
        measurementSet ZPrev ;
        ParticleSLAM particlesPreMerge(particles) ;
        ConstantVelocityState expectedPose ;
        gaussianMixture expectedMap ;
        REAL nEff ;
        timeval start, stop ;
        cout << "STARTING SIMULATION" << endl ;
        for (int n = 0 ; n < nSteps ; n++ )
        {
                gettimeofday( &start, NULL ) ;
                cout << "****** Time Step [" << n << "/" << nSteps << "] ******" << endl ;
//                if ( n == 113 )
//                        breakUpdate = true ;
//                else
//                        breakUpdate = false ;
                ZZ = allMeasurements[n] ;
                cout << "Performing prediction" << endl ;
                phdPredict(particles) ;
                if (ZPrev.size() > 0 )
                {
                        cout << "Adding birth terms" << endl ;
                        addBirths(particles,ZPrev) ;
                }
                if ( ZZ.size() > 0 )
                {
                        cout << "Performing PHD Update" << endl ;
                        particlesPreMerge = phdUpdate(particles, ZZ) ;
                }
                nEff = 0 ;
                for ( int i = 0; i < config.nParticles ; i++)
                        nEff += particles.weights[i]*particles.weights[i] ;
                nEff = 1.0/nEff/config.nParticles ;
                DEBUG_VAL(nEff) ;
                if (nEff <= config.resampleThresh )
                {
                        DEBUG_MSG("Resampling particles") ;
                        particles = resampleParticles(particles) ;
                }
                recoverSlamState(particles, &expectedPose, &expectedMap ) ;
                ZPrev = ZZ ;
                gettimeofday( &stop, NULL ) ;
                double elapsed = (stop.tv_sec - start.tv_sec)*1000 ;
                elapsed += (stop.tv_usec - start.tv_usec)/1000 ;
                fstream timeFile("loopTime.log", fstream::out|fstream::app ) ;
                timeFile << elapsed << endl ;
                timeFile.close() ;
#ifdef DEBUG
                DEBUG_MSG( "Writing Log" ) ;
//		writeParticles(particlesPreMerge,"particlesPreMerge",n) ;
//		writeLog(particles, ZZ, expectedPose, expectedMap, n) ;
                writeLogMat(particles, ZZ, expectedPose, expectedMap, n) ;
#endif
                if ( isnan(nEff) )
                {
                        cout << "nan weights detected! exiting..." << endl ;
                        break ;
                }
                for ( int i =0 ; i < config.nParticles ; i++ )
                {
                        for ( int j = 0 ; j < (int)particles.maps[i].size() ; j++ )
                        {
                                if ( particles.maps[i][j].weight == 0 )
                                {
                                        DEBUG_MSG("Invalid features detected!") ;
                                        exit(1) ;
                                }
                        }
                }
        }
#ifdef DEBUG
        string command("mv *.mat ") ;
        command += timestamp ;
        system( command.c_str() ) ;
        command = "mv *.log " ;
        command += timestamp ;
        system( command.c_str() ) ;
#endif
//	cudaPrintfEnd() ;
        cout << "DONE!" << endl ;
        return 0 ;
}
