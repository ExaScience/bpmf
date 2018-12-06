ROOT=../../../c++

CXX=mpicxx -Xpreprocessor -fopenmp
CFLAGS+=-DBPMF_MPI_ISENDIRECV_COMM -DBPMF_PROFILING
LDFLAGS+=-L/usr/local/lib -lomp

include ../../Makefile.common
