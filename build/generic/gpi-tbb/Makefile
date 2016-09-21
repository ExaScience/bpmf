ROOT=../..

CXX=g++-4.9 
CFLAGS+=-DBPMF_GPI_COMM  -DBPMF_TBB_SCHED -DBPMF_PROFILING

# for tbb
CFLAGS+=-I/usr/local/include
LDFLAGS+=-L/usr/local/lib -ltbb

# for GPI
LDLIBS+= -lGPI2 -libverbs -lpthread

include ../Makefile.common
