# Setup SPTAG on Ubuntu

In this section, will describe how to setup SPTAG on Ubuntu machine.

1. Update ubuntu and build env
```
sudo apt-get update && sudo apt-get install build-essential
```
2. Install cmake 

- Download cmake. You can find the cmake releases [here](https://github.com/Kitware/CMake/releases).
```bash 
# e.g. download cmake version 3.14.7
wget https://github.com/Kitware/CMake/releases/download/v3.14.7/cmake-3.14.7-Linux-x86_64.sh -P opt/
```
- Follow these instructions:
    1. `chmod +x opt/cmake-3.<your_version>.sh` (chmod makes the script executable)
    2. `sudo bash opt/cmake-3.<your_version>.sh` 
    3. The script installs to a target directory so in order to get the `cmake` command, make a symbolic link from the target directory where cmake was extracted to: `sudo ln -s <target_directory>/cmake-3.<your_version>/bin/* /usr/local/bin`
    4. `cmake --version` Note: If you encounter this error: *The program 'cmake' is currently not installed*, Please try the command from step 3 again with a full path (e.g. `sudo ln -s /home/<your name>/SPTAG/opt/cmake-3.<your_version>-Linux-x86_64/bin/* /usr/local/bin`)

5. Install boost
- Download boost 1.67 version:
```bash
wget https://netix.dl.sourceforge.net/project/boost/boost/1.67.0/boost_1_67_0.tar.gz
```
(There are some [version mis-matching issues](https://github.com/microsoft/SPTAG/issues/26) and reported on github issue)

6. Extract and install
```bash
tar -xzvf boost*
cd boost_1_67_0
./bootstrap.sh --prefix=/usr/local
./b2
sudo ./b2 install
sudo apt-get install swig
```

7. Generate a Release folder in the code directory which will contain all the build targets:
```bash
mkdir build
cd build && cmake .. && make
```

8. Add SPTAG module to python path
```bash
# so python can find the SPTAG module
ENV PYTHONPATH=/app/Release
```
