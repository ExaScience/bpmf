
TOPDIR=..
ROOT=$(TOPDIR)/c++

include $(TOPDIR)/Makefile.common

MCXXFLAGS=-K --ompss-2 -Wno-deprecated-declarations -DOMPSS

OBJS = mvnormal.o bpmf.o sample.o counters.o io.o gzstream.o 
HRDS = $(ROOT)/bpmf.h $(ROOT)/counters.h $(ROOT)/error.h $(ROOT)/gzstream.h $(ROOT)/io.h

mcc_ompss.o: $(ROOT)/ompss.cpp $(TOPDIR)/Makefile.common Makefile $(HRDS)
	mcxx -c $(MCXXFLAGS) $(CXXFLAGS) $< -o $@

%.o: $(ROOT)/%.cpp $(TOPDIR)/Makefile.common Makefile $(HDRS)
	g++ -c $(CXXFLAGS) $< -o $@

mcc_bpmf: $(OBJS) mcc_ompss.o
	mcxx $^ $(MCXXFLAGS) $(LDFLAGS) -o $@

bpmf: $(OBJS) ompss.o
	g++ $^ $(LDFLAGS) -o $@

clean:
	rm -f */*.o *.o */*.d *.d
	rm -f bpmf
	rm -f mcxx_*.cpp
