#!/bin/bash

rm ./*.mat
rm ./*.log
cuda-gdb --args ./bin/cuda-PHDSLAM cfg/config.cfg | tee consoleout.log
notify-send -u normal -t 3000 -i info 'done' 'Simulation done'
#beep
