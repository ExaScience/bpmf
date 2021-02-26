#!/bin/sh

set -x

LIB_SRC="../c++/bpmf.cpp ../c++/counters.cpp ../c++/gzstream.cpp ../c++/io.cpp ../c++/mvnormal.cpp ../c++/sample.cpp"
CXX_FLAGS="-std=c++11 -DBPMF_NUMLATENT=16 -Wno-deprecated-declarations -I$EBROOTEIGEN/include"

CXX=g++
$CXX -c $CXX_FLAGS $LIB_SRC

MCXX=mcxx
$MCXX -c $CXX_FLAGS ../c++/ompss.cpp

#g++ -DBPMF_NUMLATENT=16 -Wno-deprecated-declarations mcxx_ompss.cpp ../c++/bpmf.cpp ../c++/counters.cpp ../c++/gzstream.cpp ../c++/io.cpp ../c++/mvnormal.cpp ../c++/sample.cpp  -lz 2>&1 | less
