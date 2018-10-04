ROOT=../../../c++

CXX=armclang++
CFLAGS+=-DBPMF_HYBRID_COMM -DBPMF_GPI_COMM -DBPMF_OMP_SCHED -DBPMF_PROFILING

# for EIGEN3
CFLAGS+=$(EIGEN3_INCL)

# for openmp
CFLAGS+=-fopenmp

# for MPI
MPIROOT=/apps/openmpi/1.10.2
CFLAGS+=-I$(MPIROOT)/include
LDFLAGS+=-Wl,-rpath=$(MPIROOT)/lib 
LDFLAGS+=-L$(MPIROOT)/lib -lmpi

# for GPI2
GPIROOT=$(HOME)/euroexa/unimem-gpi
CFLAGS+=-I$(GPIROOT)/include
LDFLAGS+=-L$(GPIROOT)/lib64 -lGPI2

include ../../Makefile.common

