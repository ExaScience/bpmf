DEBUG?=0

ifeq ($(DEBUG),1)
CXXFLAGS=-DDEBUG -Wall -O0 -g -std=c++11 `pkg-config --cflags eigen3`
else
CXXFLAGS=-Wall -Wno-deprecated-declarations -O3 -g -std=c++11 -I. 
endif

LDFLAGS+=-ltbb

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
