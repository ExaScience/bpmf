#!/bin/bash

cd build/generic/nocomm-omp

export CPATH=$CPATH:$PREFIX/include/eigen3

sh ../../multilatent.sh

install bpmf-* $PREFIX/bin
