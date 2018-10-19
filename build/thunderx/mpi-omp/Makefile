ROOT=../../../c++

CXX=armclang++
CFLAGS+=-DBPMF_MPI_ISENDIRECV_COMM -DBPMF_OMP_SCHED -DBPMF_PROFILING

# for EIGEN3
CFLAGS+=$(EIGEN3_INCL)

# for openmp
CFLAGS+=-fopenmp

# for MPI
MPIROOT=/apps/mpich/3.2
CFLAGS+=-I$(MPIROOT)/include
LDFLAGS+=-Wl,-rpath=$(MPIROOT)/lib 
LDFLAGS+=-L$(MPIROOT)/lib -lmpi

# for GPI2
# GPIROOT=$(HOME)/euroexa/unimem-gpi
# CFLAGS+=-I$(GPIROOT)/include
# LDFLAGS+=-L$(GPIROOT)/lib64 -lGPI2-dbg

LDFLAGS+=-Wl,-rpath,/apps/ARM/hpcCompiler/18.3/lib
LDFLAGS+=-Wl,-rpath,/apps/gcc/7.1.0/lib64

include ../../Makefile.common

