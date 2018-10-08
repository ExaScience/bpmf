#!/bin/bash

BASEDIR=$1

cd $BASEDIR

BPMF=gpi-omp/bpmf

ARGS="-n $BASEDIR/data/movielens/ml-train.mtx -p $BASEDIR/data/movielesn/ml-test.mtx"

ldd $BPMF

$BPMF $ARGS
