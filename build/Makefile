
TOPDIR=..
ROOT=$(TOPDIR)/c++

include $(TOPDIR)/Makefile.common

MCXXFLAGS=--ompss-2 -k -Wno-deprecated-declarations

ompss.o: $(ROOT)/ompss.cpp $(TOPDIR)/Makefile.common Makefile
	mcxx -c $(MCXXFLAGS) $(CXXFLAGS) $< -o $@

%.o: $(ROOT)/%.cpp $(TOPDIR)/Makefile.common Makefile
	g++ -c $(CXXFLAGS) $< -o $@

bpmf: mvnormal.o bpmf.o sample.o counters.o io.o gzstream.o ompss.o
	mcxx $^ $(MCXXFLAGS) $(LDFLAGS) -o $@

clean:
	rm -f */*.o *.o */*.d *.d
	rm -f bpmf

