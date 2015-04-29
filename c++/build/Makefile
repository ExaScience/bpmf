
CXX=g++
# CXXFLAGS=-DDEBUG -O0 -g -std=c++11 `pkg-config --cflags eigen3`
CXXFLAGS=-O3 -g -std=c++11 `pkg-config --cflags eigen3`
VPATH=..
LINK.o=$(CXX) $(CXXFLAGS)

bpmf: mvnormal.o bpmf.o

clean:
	rm bpmf *.o


