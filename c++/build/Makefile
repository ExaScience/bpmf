DEBUG?=0

#CXX=icc -gcc-name=gcc-4.7
CXX=g++-4.9

ifeq ($(DEBUG),1)
CXXFLAGS=-DDEBUG -Wall -O0 -g -std=c++11
else
CXXFLAGS=-DNDEBUG -Wall -O3 -g -std=c++11 
endif

# for tbb
# LDFLAGS+=-ltbb

# for openmp
CXXFLAGS+=-fopenmp -DBREAKPOINT=$(BREAKPOINT)

# Add Eigen path here
CPPFLAGS=-I/usr/local/include/eigen3

VPATH=..
LINK.o=$(CXX) $(CXXFLAGS) $(LDFLAGS)

bpmf: mvnormal.o bpmf.o

clean:
	rm -f bpmf *.o *_test *_bench

mvnormal_test: mvnormal.cpp
	$(CXX) -DTEST_MVNORMAL $(CXXFLAGS) -o $@ $?

mvnormal_bench: mvnormal.cpp
	$(CXX) -DBENCH_MVNORMAL $(CXXFLAGS) -o $@ $?

chol_bench: mvnormal.cpp
	$(CXX) -DBENCH_CHOL $(CXXFLAGS) -o $@ $?

sample_test: bpmf.cpp
	$(CXX) -DTEST_SAMPLE $(CXXFLAGS) -o $@ $?

LOG=bpmf-chmbl19_360-$(shell date "+%Y-%m-%d_%H%M").log
REF=bpmf-chmbl19_360-ref.log
check: bpmf

	./bpmf chembl-IC50-360targets_probe.mtx chembl-IC50-360targets_sample.mtx 2>&1 | grep ^Iteration | cut -f 1-5 | head -n 20 >$(LOG)
	diff -u $(LOG) $(REF)
