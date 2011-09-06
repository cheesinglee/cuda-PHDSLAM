#!/bin/bash

rm ./*.mat
rm ./*.log
rm ./bin/cuda-PHDSLAM
make clean
make -w
time ./bin/cuda-PHDSLAM cfg/config.cfg | tee consoleout.log
notify-send -u normal -t 3000 'done' 'Simulation Finished'

