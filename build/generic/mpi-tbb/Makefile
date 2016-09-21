ROOT=../../../c++

CXX=mpicxx
CFLAGS+=-DBPMF_MPI_ISENDIRECV_COMM  -DBPMF_TBB_SCHED -DBPMF_PROFILING

# for tbb
CFLAGS+=-I/usr/local/include
LDFLAGS+=-L/usr/local/lib -ltbb


include ../../Makefile.common
