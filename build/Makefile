
TOPDIR=..
ROOT=$(TOPDIR)/c++

include $(TOPDIR)/Makefile.common

MCXXFLAGS=--ompss-2 -k -Wno-deprecated-declarations

OBJS = mvnormal.o bpmf.o sample.o counters.o io.o gzstream.o ompss.o
HRDS = $(ROOT)/bpmf.h $(ROOT)/counters.h $(ROOT)/error.h $(ROOT)/gzstream.h $(ROOT)/io.h

ompss.o: $(ROOT)/ompss.cpp $(TOPDIR)/Makefile.common Makefile $(HRDS)
	mcxx -c $(MCXXFLAGS) $(CXXFLAGS) $< -o $@

%.o: $(ROOT)/%.cpp $(TOPDIR)/Makefile.common Makefile $(HDRS)
	g++ -c $(CXXFLAGS) $< -o $@


bpmf: $(OBJS)
	mcxx $^ $(MCXXFLAGS) $(LDFLAGS) -o $@

clean:
	rm -f */*.o *.o */*.d *.d
	rm -f bpmf
	rm -f mcxx_*.cpp
