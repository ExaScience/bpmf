#!/bin/sh

set -e

NL="8 16 32 64 128 10 20 30 40 50 60 70 80 90 100"

for K in $NL
do
    BUILD_DIR=build-$K
    if [ -d $BUILD_DIR ]
    then
        cd $BUILD_DIR
    else
        mkdir $BUILD_DIR
        cd $BUILD_DIR
        cmake .. -DBPMF_NUMLATENT=$K $*
    fi
    make -j
    cp bpmf $CONDA_PREFIX/bin/bpmf-$K
    cd ..
done
