ROOT=../../../bpmf/c++

CXX=g++
CFLAGS+=-DBPMF_MPI_ISENDIRECV_COMM  -DBPMF_TBB_SCHED -DBPMF_PROFILING

SUPPORT=/work/y14/shared/eurompi/support

# for MPICH
CFLAGS+=-I$(CRAY_MPICH_DIR)/include
LDFLAGS+=-Wl,-rpath=$(CRAY_MPICH_DIR)/lib 
LDFLAGS+=-L$(CRAY_MPICH_DIR)/lib -lmpich

# for tbb
CFLAGS+=-I$(SUPPORT)/tbb/include
LDFLAGS+=-Wl,-rpath=$(SUPPORT)/tbb/lib 
LDFLAGS+=-L$(SUPPORT)/tbb/lib -ltbb

include ../../Makefile.common

