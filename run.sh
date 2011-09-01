#!/bin/bash

rm ./*.mat
rm ./*.log
./bin/cuda-PHDSLAM cfg/config.ini | tee consoleout.log
beep
