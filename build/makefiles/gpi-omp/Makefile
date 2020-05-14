ROOT=../../../c++

include ../Makefile.common

CXX=mpiicpc
CFLAGS+=-DBPMF_HYBRID_COMM -DBPMF_OMP_SCHED -DBPMF_PROFILING

# for openmp
CFLAGS+=-fopenmp -I$(HOME)/epeec/GPI-2/1.4.0/include
 
# for GPI2
LDFLAGS+=-L$(HOME)/epeec/GPI-2/1.4.0/lib64 -lGPI2 -libverbs

