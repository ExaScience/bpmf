FROM ubuntu:16.04

RUN apt-get update && \
   apt-get install -y --no-install-recommends software-properties-common && \
   add-apt-repository ppa:lkoppel/robotics

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
       git g++ python3 \
       gdb \
       vim \
       libblas-dev \
       liblapack-dev \
       liblapacke-dev \
       libopenmpi-dev \
       openmpi-bin \
       libeigen3-dev \
       libboost-all-dev \
       ca-certificates \
       wget \
       libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

ARG CMAKE_VERSION=3.17.0

RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.sh  && \
   sh cmake-${CMAKE_VERSION}-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir && \
   rm cmake-${CMAKE_VERSION}-Linux-x86_64.sh


RUN wget -O HighFive.tar.gz https://github.com/BlueBrain/HighFive/archive/v2.2.tar.gz && \
    tar xzf HighFive.tar.gz && \
    cd HighFive* && mkdir build && cd build && \
    cmake .. && \
    make -j2 && make install && \
    cd ../../ && rm -r HighFive* 

RUN adduser --disabled-password --gecos "Ubuntu User" ubuntu
USER ubuntu

