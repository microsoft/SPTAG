## Requirements

### swig >= 3.0

1. download windows binaries at https://sourceforge.net/projects/swig/files/swigwin/ and unzip content 
2. add the folder path (where swig.exe is located) to PATH environment variable, for instance `C:\Sptag\swigwin-3.0.12\`

### cmake >= 3.12.0

1. we recommend to install Visual Studio 2019 (will be needed to build components)
2. cmake.exe is already installed with Visual Studio 2019 when you select `Desktop Development with C++`, just ensure this option is enabled in the installation workloads

![Visual Studio](docs/img/visualstudio.png)

### boost == 1.67.0

easiest solution is to install boost C++ libraries from https://sourceforge.net/projects/boost/files/boost-binaries/1.67.0/, it avoids to compile yourself all libraries. 

1. select last version for 1.67.0 (boost_1_67_0-msvc-14.1-64.exe)
2. launch installation
3. add the folder path to PATH environment variable, for instance `C:\Sptag\boost_1_67_0\`


## Build

1. open `Developer Command Prompt for VS 2019`
2. clone [microsoft/SPTAG](https://github.com/microsoft/SPTAG/)
3. go the folder location where you cloned the repo
4. execute the following commands
```
mkdir build
cd build
cmake -A x64 ..
```
5. in the build folder, open SPTAGLib.sln solution in Visual Studio
6. compile all projects

[..]/build/release contains now all components needed, add this path to PYTHONPATH environment variable to reference modules