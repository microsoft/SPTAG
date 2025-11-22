FROM ubuntu:22.04
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# Update and install basic dependencies
RUN apt-get update && apt-get -y install wget build-essential swig cmake git libnuma-dev python3-dev python3-distutils \
    python3-pip software-properties-common

# Ubuntu 22.04 comes with Boost 1.74 which should work, but let's ensure we have all required components
# Including the development headers for Beast (HTTP/WebSocket library)
RUN apt-get -y install \
    libboost-all-dev \
    libboost-filesystem-dev \
    libboost-test-dev \
    libboost-serialization-dev \
    libboost-regex-dev \
    libboost-thread-dev \
    libboost-system-dev \
    libboost-chrono-dev \
    libboost-date-time-dev \
    libboost-atomic-dev \
    libboost-context-dev \
    libboost-coroutine-dev \
    libtbb-dev

# Install Python dependencies
RUN python3 -m pip install numpy

ENV PYTHONPATH=/app/Release

# Copy project files
COPY CMakeLists.txt ./
COPY AnnService ./AnnService/
COPY Test ./Test/
COPY Wrappers ./Wrappers/
COPY GPUSupport ./GPUSupport/
COPY base ./base/
COPY build_murren_linux.ini ./

# Build with C++17 support for filesystem and proper Boost configuration
RUN mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_STANDARD=17 \
          -DSPDK=OFF \
          -DROCKSDB=OFF \
          -DTBB=OFF \
          .. && \
    make -j$(nproc) && \
    cd ..

RUN mkdir -p /app/base_index && \
    ./Release/indexbuilder -a SPANN -c build_murren_linux.ini -d 256 -v Int8 -f TXT -o /app/base_index -i /app/base/base_vector.tsv -t 16 -m true

# Create directories for runtime data and config
RUN mkdir -p /app/data /app/config /app/logs

# Copy configuration files
COPY AnnService.docker.ini /app/config/AnnService.ini

# Set working directory to Release folder where binaries are
WORKDIR /app/Release

# Expose both TCP socket port and HTTP port
EXPOSE 8888


HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:8888/health || exit 1

# For HTTP/Socket mode:
CMD ["./server", "-m", "http", "-c", "/app/config/AnnService.ini"]
# For debugging:
# CMD ["tail", "-f", "/dev/null"]