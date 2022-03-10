
TOPDIR=..
ROOT=$(TOPDIR)/c++

include $(TOPDIR)/Makefile.common

MCCFLAGS=-K --ompss-2 -Wno-deprecated-declarations -DOMPSS

ifeq ($(CLUSTER_BACKEND), "ARGO")
CXXFLAGS += -DCLUSTER_ARGO
endif

OBJS = mvnormal.o bpmf.o sample.o counters.o io.o gzstream.o 
HRDS = $(ROOT)/bpmf.h $(ROOT)/counters.h $(ROOT)/error.h $(ROOT)/gzstream.h $(ROOT)/io.h

all: bpmf

mcc_ompss.o: $(ROOT)/ompss.c $(TOPDIR)/Makefile.common Makefile $(HRDS)
	mcc -c $(MCCFLAGS) $(CFLAGS) $< -o $@

%.o: $(ROOT)/%.cpp $(TOPDIR)/Makefile.common Makefile $(HDRS)
	g++ -c $(CFLAGS) $(CXXFLAGS) $< -o $@

bpmf: $(OBJS) mcc_ompss.o
	mcxx $^ $(MCCFLAGS) $(CFLAGS) $(CXXFLAGS) $(LDFLAGS) -o $@

clean:
	rm -f */*.o *.o */*.d *.d
	rm -f bpmf
	rm -f mcxx_*.cpp
