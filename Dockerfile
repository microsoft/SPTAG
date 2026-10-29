FROM mcr.microsoft.com/oss/mirror/docker.io/library/ubuntu:24.04
WORKDIR /app

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build swig python3-dev \
    libboost-system-dev libboost-thread-dev libboost-serialization-dev \
    libboost-regex-dev libboost-filesystem-dev libboost-test-dev libtbb-dev libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

COPY CMakeLists.txt ./
COPY AnnService ./AnnService/
COPY Test ./Test/
COPY Wrappers ./Wrappers/
COPY GPUSupport ./GPUSupport/
COPY ThirdParty/zstd ./ThirdParty/zstd/

RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DFOLLY=OFF -DTBB=ON -DSPDK=OFF -DROCKSDB=OFF -DTIKV=OFF -DGPU=OFF -DRABITQ=OFF \
    && cmake --build build --target indexbuilder indexsearcher SPTAGTest --parallel 2
