FROM gcc:10
COPY . /work
WORKDIR /work
RUN g++ -g -fopenmp -std=c++11 -o bpmf bpmf.cpp
CMD ["./bpmf"]
