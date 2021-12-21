wget -O boost_1_67_0.tar.gz https://sourceforge.net/projects/boost/files/boost/1.67.0/boost_1_67_0.tar.gz/download
tar xzvf boost_1_67_0.tar.gz
cd boost_1_67_0/

sudo apt-get update
sudo apt-get install build-essential g++ python-dev autotools-dev libicu-dev libbz2-dev libboost-all-dev
./bootstrap.sh --prefix=/usr/
./b2
sudo ./b2 install
cd ..
mkdir build
cd build && cmake .. && sudo make install
