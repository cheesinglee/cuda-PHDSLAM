### qmake project file for cuda-PHDSLAM. Based on file found at
### http://cudaspace.wordpress.com/2011/04/07/qt-creator-cuda-linux/



# Basic .pro configuration
SOURCES += src/main.cpp \
   src/phdfilter.cu \
    src/rng.cpp \
    src/phdfilterwrapper.cpp \
    src/fastslam.cu \
    src/munkres.cu \
    src/device_math.cu
# Cuda sources
SOURCES -= src/phdfilter.cu \
         src/fastslam.cu \
        src/munkres.cu \
        src/device_math.cu
CUDA_SOURCES += src/phdfilter.cu
#            src/device_math.cu
#            src/fastslam.cu \
#            src/munkres.cu
# Project dir and outputs
PROJECT_DIR = $$system(pwd)
OBJECTS_DIR = $$PROJECT_DIR/Obj
DESTDIR = $$PROJECT_DIR/bin

QMAKE_CXXFLAGS += -DDEBUG
QMAKE_CXXFLAGS += -Wall -Wno-deprecated -fpic -DOC_NEW_STYLE_INCLUDES -fpermissive -fno-strict-aliasing

# remove qt libs
CONFIG += dll
CONFIG += link_prl
QT     -= gui core
LIBS   -= -lQtGui -lQtCore

### Pickling Tools ######################################
INCLUDEPATH += PicklingTools121Release/C++
INCLUDEPATH += PicklingTools121Release/C++/opencontainers_1_6_9/include
#########################################################

#### MATLAB external interface ###############################################
##MATLAB_PATH = /home/cheesinglee/matlab2010a/
##MATLAB_PATH = /opt/Matlab-R2010a/ # llebre
#MATLAB_PATH = /opt/Matlab # kermit
#QMAKE_RPATHDIR += $$MATLAB_PATH/bin/glnxa64
#QMAKE_LIBDIR += $$MATLAB_PATH/bin/glnxa64/
#LIBS += -lmx -lmat
#INCLUDEPATH += $$MATLAB_PATH/extern/include
##############################################################################

#### Boost libraries ###################################################
LIBS += -lboost_program_options -lboost_random
######################################################################

#### FFTW libraries ######################################################
# needs to be statically linked to avoid conflicts with the MATLAB libs
LIBS += -lfftw3 -lm
##########################################################################

CUDA_LIBS = $$LIBS
CUDA_LIBS -= -lboost_program_options

# Path to cuda SDK install
CUDA_SDK = /usr/cuda/sdk/C
#CUDA_SDK = /opt/cuda/sdk/C
# Path to cuda toolkit install
CUDA_DIR = /usr/cuda/ # for my machines
#CUDA_DIR = /opt/cuda/ # for llebre
# GPU architecture
#CUDA_GENCODE = arch=compute_13,code=sm_13
CUDA_GENCODE = arch=compute_20,code=sm_20
# nvcc flags (ptxas option verbose is always useful)
CUDA_GCC_BINDIR=/opt/gcc-4.4
#CUDA_GCC_BINDIR=/usr/bin
NVCCFLAGS = --compiler-options -fno-strict-aliasing,-fpermissive --ptxas-options=-v --compiler-bindir=$$CUDA_GCC_BINDIR --linker-options -rpath=$$MATLAB_PATH/bin/glnxa64
# include paths
INCLUDEPATH += $$CUDA_DIR/include/cuda/
INCLUDEPATH += $$CUDA_DIR/include/
INCLUDEPATH += $$CUDA_SDK/common/inc/
INCLUDEPATH += $$CUDA_SDK/../shared/inc/
# lib dirs
QMAKE_LIBDIR += $$CUDA_DIR/lib64
QMAKE_LIBDIR += $$CUDA_SDK/lib
QMAKE_LIBDIR += $$CUDA_SDK/common/lib
# libs - note than i'm using a x_86_64 machine
LIBS += -lcudart -lcutil_x86_64
# join the includes in a line
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

# Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o

cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -g -G -O0 -gencode=$$CUDA_GENCODE -c $$NVCCFLAGS $$CUDA_INC $$CUDA_LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

cuda.dependcy_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -g -G -M $$CUDA_INC $$NVCCFLAGS ${QMAKE_FILE_NAME}
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda

HEADERS += \
    src/slamparams.h \
    src/slamtypes.h \
    src/rng.h \
    src/phdfilterwrapper.h \
    src/device_math.h

OTHER_FILES += \
    cfg/config.cfg

