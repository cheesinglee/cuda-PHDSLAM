#!/bin/bash

rm ./*.mat
rm ./*.log
rm ./bin/cuda-PHDSLAM
qmake
make clean
make -w
cuda-gdb --args ./bin/cuda-PHDSLAM cfg/config.cfg
notify-send -u normal -t 3000 'done' 'Simulation Finished'

#beep
