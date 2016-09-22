ROOT=../../../c++

CXX=g++
CFLAGS+=-DBPMF_NO_COMM  -DBPMF_TBB_SCHED -DBPMF_PROFILING

SUPPORT=/work/y14/shared/eurompi/support

# for tbb
CFLAGS+=-I$(SUPPORT)/tbb/include
LDFLAGS+=-Wl,-rpath=$(SUPPORT)/tbb/lib 
LDFLAGS+=-L$(SUPPORT)/tbb/lib -ltbb

include ../../Makefile.common

