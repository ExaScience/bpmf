CXX=g++-4.9
DEBUG?=0

ifeq ($(DEBUG),1)
CXXFLAGS=-DDEBUG -O0 -g -std=c++11 `pkg-config --cflags eigen3`
else
CXXFLAGS=-DNDEBUG -O3 -g -std=c++11 `pkg-config --cflags eigen3` # -fopenmp
endif

VPATH=..
LINK.o=$(CXX) $(CXXFLAGS)

bpmf: mvnormal.o bpmf.o

clean:
	rm -f bpmf *.o *_test

mvnormal_test: mvnormal.cpp
	$(CXX) -DTEST_MVNORMAL $(CXXFLAGS) -o $@ $?

sample_test: bpmf.cpp
	$(CXX) -DTEST_SAMPLE $(CXXFLAGS) -o $@ $?
