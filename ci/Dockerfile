FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
       git g++ cmake python3 \
       zlib1g-dev \
       libblas-dev \
       liblapack-dev \
       liblapacke-dev \
       libopenmpi-dev \
       openmpi-bin \
       libeigen3-dev \
       openssh-client \
    && rm -rf /var/lib/apt/lists/*