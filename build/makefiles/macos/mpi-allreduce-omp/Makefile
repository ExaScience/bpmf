ROOT=../../../c++

CXX=mpicxx 
MPIRUN=mpirun

include ../Makefile.common

CFLAGS+=-Xpreprocessor -fopenmp -g
CFLAGS+=-DBPMF_MPI_ALLREDUCE_COMM -DBPMF_PROFILING
LDFLAGS+=-lomp
