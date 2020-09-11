#!/bin/bash

cd build/makefiles/nocomm-omp

export CPATH=$CPATH:$PREFIX/include/eigen3

sh ../../multilatent.sh

install bpmf-* $PREFIX/bin
