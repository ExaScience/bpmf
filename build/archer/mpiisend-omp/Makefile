ROOT=../../../c++

CXX=g++ -fopenmp
CFLAGS+=-DBPMF_MPI_ISENDIRECV_COMM  -DBPMF_OMP_SCHED -DBPMF_PROFILING

# for MPICH
CFLAGS+=-I$(CRAY_MPICH_DIR)/include
LDFLAGS+=-Wl,-rpath=$(CRAY_MPICH_DIR)/lib 
LDFLAGS+=-L$(CRAY_MPICH_DIR)/lib -lmpich

include ../../Makefile.common

