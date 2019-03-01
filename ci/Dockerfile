FROM ubuntu:18.04

RUN apt-get update && \
   apt-get install -y software-properties-common

RUN add-apt-repository ppa:lkoppel/robotics && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
       git g++ cmake python3 \
       libblas-dev \
       liblapack-dev \
       liblapacke-dev \
       libopenmpi-dev \
       openmpi-bin \
       libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*