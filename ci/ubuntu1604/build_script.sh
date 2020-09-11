#!/bin/sh
#
# Start this script in a Docker container like this:
#
#  docker run -eCPU_COUNT=2 -v $(git rev-parse --show-toplevel):/bpmf -ti ubuntu1604 /bpmf/ci/ubuntu1604/build_script.sh
#
# where ubuntu1604 is the image name


set -e
set -x

BUILD_DIR=$HOME/build

rm -rf $BUILD_DIR  && mkdir $BUILD_DIR && cd $BUILD_DIR
cmake /bpmf -DBPMF_COMM=MPI_ALLREDUCE_COMM -DENABLE_OPENMP=OFF -DENABLE_REDUCE=OFF
make -j${CPU_COUNT}
ctest -VV