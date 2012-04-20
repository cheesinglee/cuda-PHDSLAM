/*
 * slamparams.h
 *
 *  Created on: Apr 11, 2011
 *      Author: cheesinglee
 */

#ifndef SLAMPARAMS_H_
#define SLAMPARAMS_H_

#include <math.h>

#ifndef M_PI
#define M_PI 3.1415926535897932385
#endif

#define DEG2RAD(x) ((x)*M_PI/180)
#define NPARTICLES 500
#define INITPX 30
#define INITPY 5
#define INITPTHETA 0
#define INITVX 7
#define INITVY 0
#define INITVTHETA DEG2RAD(18)
#define STDX 0.5
#define STDY 0
#define STDTHETA DEG2RAD(0.5)

#define DT 0.1
#define DT2 DT*DT

#define MAXRANGE 18
#define MAXBEARING M_PI
#define INFLATENOISE 1
#define STDRANGE 1
#define STDBEARING 0.0524
#define CLUTTERRATE 15
#define PD 0.98
#define VARRANGE STDRANGE*STDRANGE*INFLATENOISE
#define VARBEARING STDBEARING*STDBEARING*INFLATENOISE
#define CLUTTERDENSITY CLUTTERRATE/( MAXRANGE*MAXRANGE*MAXBEARING )

#define BIRTH_COMPAT_THRESHOLD 4.6052

#define MAXMEASUREMENTS 256

#define BIRTHWEIGHT 0.05

#define MINGAUSSIANWEIGHT 0.00001
#define MINSEPARATION 5
#define MAXGAUSSIANS 100

#define RESAMPLETHRESHOLD 0.75

#define LOGFILENAME "cudaphdslam.log"

#endif /* SLAMPARAMS_H_ */