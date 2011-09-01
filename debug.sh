#!/bin/bash

rm ./*.mat
rm ./*.log
cuda-gdb --args ./bin/cuda-PHDSLAM cfg/config.ini | tee consoleout.log
#beep
