FROM mcr.microsoft.com/oss/mirror/docker.io/library/ubuntu:20.04
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y install wget build-essential \
    swig cmake git \
    libboost-filesystem-dev libboost-test-dev libboost-serialization-dev libboost-regex-dev libboost-serialization-dev libboost-regex-dev libboost-thread-dev libboost-system-dev

ENV PYTHONPATH=/app/Release

COPY CMakeLists.txt ./
COPY AnnService ./AnnService/
COPY Test ./Test/
COPY Wrappers ./Wrappers/
COPY GPUSupport ./GPUSupport/

# install zstd
COPY ThirdParty ./ThirdParty/
RUN cd ThirdParty/zstd/build/cmake && rm -rf builddir && mkdir builddir && cd builddir && cmake .. && make -j$(nproc) && make install

RUN mkdir build && cd build && cmake .. && make -j$(nproc)
