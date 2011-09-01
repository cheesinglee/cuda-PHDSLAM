#!/bin/bash

rm ./*.mat
rm ./*.log
rm ./bin/cuda-PHDSLAM
make clean
make -w
time ./bin/cuda-PHDSLAM cfg/config.ini | tee consoleout.log
beep
