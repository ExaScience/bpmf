#!/bin/sh

set -e

rm -rf output
mkdir output

mpirun -np 4 ../../build/generic/mpi-omp/bpmf -k -i 9 -b 0 -v -n train.mtx -p test.mtx -o output/ -t 1
# ../../build/generic/nocomm-omp/bpmf -k -i 9 -b 0 -v -n train.mtx -p test.mtx -o output/ -t 1

python compute_mu_lambda.py

