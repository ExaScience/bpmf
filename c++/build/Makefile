DEBUG?=0

#CXX=icc -gcc-name=gcc-4.7
CXX=g++-4.8

ifeq ($(DEBUG),1)
CXXFLAGS=-DDEBUG -Wall -O0 -g -std=c++11
else
CXXFLAGS=-DNDEBUG -Wall -O3 -g -std=c++11 
endif

# for tbb
# LDFLAGS+=-ltbb

# for openmp
CXXFLAGS+=-fopenmp

# Add Eigen path here
CPPFLAGS=-I /home/tomh/usr/include/eigen3

VPATH=..
LINK.o=$(CXX) $(CXXFLAGS) $(LDFLAGS)

bpmf: mvnormal.o bpmf.o

clean:
	rm -f bpmf *.o *_test *_bench

mvnormal_test: mvnormal.cpp
	$(CXX) -DTEST_MVNORMAL $(CXXFLAGS) -o $@ $?

mvnormal_bench: mvnormal.cpp
	$(CXX) -DBENCH_MVNORMAL $(CXXFLAGS) -o $@ $?

sample_test: bpmf.cpp
	$(CXX) -DTEST_SAMPLE $(CXXFLAGS) -o $@ $?
