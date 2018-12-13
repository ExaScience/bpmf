ROOT=../../../c++

CXX=g++
CFLAGS+=-DBPMF_HYBRID_COMM -DBPMF_OMP_SCHED -DBPMF_PROFILING

# for openmp
CFLAGS+=-fopenmp

# for GPI2
GPIROOT=$(HOME)/euroexa/unimem-gpi
CFLAGS+=-I$(GPIROOT)/include
LDFLAGS+=-L$(GPIROOT)/lib64 -lGPI2

include ../../Makefile.common

