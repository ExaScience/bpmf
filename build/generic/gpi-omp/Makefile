ROOT=../../../c++

CXX=g++
CFLAGS+=-DBPMF_GPI_COMM -DBPMF_OMP_SCHED -DBPMF_PROFILING

# for EIGEN3
CFLAGS+=-I$(EBROOTEIGEN3)/include

# for openmp
CFLAGS+=-fopenmp

# for GPI2
GPI2=$(HOME)/tmp/GPI-2-1.3.0
CFLAGS+=-I$(GPI2)/include
LDFLAGS+=-L$(GPI2)/lib64 -lGPI2-dbg -libverbs

# rpath
LDFLAGS+=-Wl,-rpath,$(EBROOTGCC)/lib64

include ../../Makefile.common

