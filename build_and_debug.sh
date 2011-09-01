#!/bin/bash

rm ./*.mat
rm ./*.log
rm ./bin/cuda-PHDSLAM
make clean
make -w
cuda-gdb --args ./bin/cuda-PHDSLAM cfg/config.ini
#beep
