
TOPDIR=..
ROOT=$(TOPDIR)/c++

ifndef BPMF_NUMLATENT
	BPMF_NUMLATENT=32
endif

#CXXFLAGS=$(CFLAGS) -std=c++0x #-cxxlib=/opt/gcc/6.3.0/snos/lib64 # Intel Compiler
CXXFLAGS=$(CFLAGS) -std=c++11   # original line

CXXFLAGS+=-Wall -Wextra -Wfatal-errors -I$(ROOT)
CXXFLAGS+=-Wno-unknown-pragmas
CXXFLAGS+=-Wno-unused-local-typedefs
CXXFLAGS+=-I/usr/include/eigen3
CXXFLAGS+=-DEIGEN_DONT_PARALLELIZE
CXXFLAGS+=-DBPMF_NUMLATENT=$(BPMF_NUMLATENT)

#CXXFLAGS+=-w -O3 -g -DNDEBUG
#CXXFLAGS+=-no-inline-max-size -no-inline-max-total-size -O3 -g -DNDEBUG # -w=1 #-axSSSE3 # Intel Compiler
#CXXFLAGS+=-ffast-math -O3 -g -DNDEBUG  # original line
#CXXFLAGS+=-fopt-info-optimized=gnu_fopt_info_optimized.txt -O3 -g -DNDEBUG -ffast-math #-fopt-info-vec #-ftree-loop-optimize # Report for GNU
#CXXFLAGS+=-ffast-math -O3 -g -DNDEBUG # report for Cray Compiler

CXXFLAGS+=-O2 -g 
#CXXFLAGS+=-O0 -g

MCXXFLAGS=--ompss-2

LDFLAGS=-lz

LINK.o=$(CXX) $(CXXFLAGS) $(LDFLAGS) $(TARGET_ARCH)
OUTPUT_OPTION=-MMD -MP -o $@

.PHONY: all clean test

all: bpmf

ompss.o: $(ROOT)/ompss.cpp $(TOPDIR)/Makefile
	mcxx -c $(MCXXFLAGS) $(CXXFLAGS) $< -o $@

%.o: $(ROOT)/%.cpp $(TOPDIR)/Makefile
	g++ -c $(CXXFLAGS) $< -o $@

bpmf: mvnormal.o bpmf.o sample.o counters.o io.o gzstream.o ompss.o
	mcxx $^ $(MCXXFLAGS) $(LDFLAGS) -o $@

clean:
	rm -f */*.o *.o */*.d *.d
	rm -f bpmf

run: bpmf
	$(MPIRUN) ./bpmf -i 4 -n $(TOPDIR)/data/movielens/ml-train.mtx -p $(TOPDIR)/data/movielens/ml-test.mtx
	$(MPIRUN) ./bpmf -i 4 -n $(TOPDIR)/data/movielens/ml-train.mtx.gz -p $(TOPDIR)/data/movielens/ml-test.mtx.gz

check: run
	cd $(TOPDIR)/data/tiny && ./run_test.sh $(PWD)/bpmf 4.1

install: bpmf
	install bpmf $(PREFIX)/bin

# DFILES 
DFILES=$(CCFILES:.cpp=.d)
DFILES=$(wildcard */*.d *.d)

-include $(DFILES)