ROOT=../../../c++

CXX=mpicxx
CFLAGS+=-DBPMF_GPI_COMM -DBPMF_OMP_SCHED -DBPMF_PROFILING

# for EIGEN3
CFLAGS+=-I$(EBROOTEIGENG3)/include

# for openmp
CFLAGS+=-fopenmp

# for GPI2
CFLAGS+=-I$(EBROOTGPI2)/include
LDFLAGS+=-L$(EBROOTGPI2)/lib64 -lGPI2 -libverbs

include ../../Makefile.common

