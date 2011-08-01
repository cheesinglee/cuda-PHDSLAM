### qmake project file for cuda-PHDSLAM. Based on file found at
### http://cudaspace.wordpress.com/2011/04/07/qt-creator-cuda-linux/



# Basic .pro configuration
SOURCES += src/main.cpp \
   src/phdfilter.cu \
    src/rng.cpp
# Cuda sources
SOURCES -= src/phdfilter.cu
CUDA_SOURCES += src/phdfilter.cu
# Project dir and outputs
PROJECT_DIR = $$system(pwd)
OBJECTS_DIR = $$PROJECT_DIR/Obj
DESTDIR = $$PROJECT_DIR/bin

QMAKE_CFLAGS += -DDEBUG

### MATLAB external interface ###############################################
QMAKE_RPATHDIR += /home/cheesinglee/matlab2010a/bin/glnxa64
LIBS += -L/home/cheesinglee/matlab2010a/bin/glnxa64/ -lmx -lmat
INCLUDEPATH += /home/cheesinglee/matlab2010a/extern/include
#############################################################################

#### Boost libraries ###################################################
LIBS += -lboost_program_options -lboost_random

CUDA_LIBS = $$LIBS
CUDA_LIBS -= -lboost_program_options

# Path to cuda SDK install
CUDA_SDK = /usr/share/cuda-sdk/C
# Path to cuda toolkit install
CUDA_DIR = /usr/
# GPU architecture
CUDA_ARCH = sm_20
# nvcc flags (ptxas option verbose is always useful)
NVCCFLAGS = --compiler-options -fno-strict-aliasing,-fpermissive --ptxas-options=-v --compiler-bindir=/opt/gcc-4.4 --linker-options -rpath=/home/cheesinglee/matlab2010a/bin/glnxa64
# include paths
INCLUDEPATH += $$CUDA_DIR/include/cuda/
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

cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -g -G -arch=$$CUDA_ARCH -c $$NVCCFLAGS $$CUDA_INC $$CUDA_LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

cuda.dependcy_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -g -G -M $$CUDA_INC $$NVCCFLAGS ${QMAKE_FILE_NAME}
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda

HEADERS += \
    src/slamparams.h \
    src/slamtypes.h \
    src/rng.h

OTHER_FILES += \
    cfg/config.ini
