#!/bin/sh


rm -rf output
mkdir output

../../build/generic/nocomm-omp/bpmf -i 4 -b 0 -v -n train.mtx -p test.mtx -o output/ -t 1

