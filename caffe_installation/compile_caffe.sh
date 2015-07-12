# To install caffe and pycaffe on Ubuntu 14.04 x64 (also tested on Kubunty 14.10 x64). CPU only.
# TODO: define number of CPUs + ROOTDIR for caffe

#http://caffe.berkeleyvision.org/install_apt.html : (general install info: http://caffe.berkeleyvision.org/installation.html)
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev
sudo apt-get install -y --no-install-recommends libboost-all-dev
sudo apt-get install -y libatlas-base-dev 
sudo apt-get install -y python-dev 
sudo apt-get install -y python-pip git

# For Ubuntu 14.04
sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler 

# lmdb
#git clone https://gitorious.org/mdb/mdb.git
#cd mdb/libraries/liblmdb
#make && make install 

git clone https://github.com/LMDB/lmdb.git
cd lmdb/libraries/liblmdb
sudo make 
sudo make install

# Pre-requisites Franck
sudo apt-get install -y cmake
sudo apt-get install -y protobuf-compiler
sudo apt-get install -y libffi-dev python-dev build-essential
sudo pip install lmdb
sudo pip install numpy
sudo apt-get install python-numpy
sudo apt-get install -y gfortran # required by scipy
sudo pip install scipy # required by scikit-image
sudo pip install scikit-image # to fix https://github.com/BVLC/caffe/issues/50

# Get caffe (http://caffe.berkeleyvision.org/installation.html#compilation)
cd
mkdir caffe
cd caffe
wget https://github.com/BVLC/caffe/archive/master.zip
unzip -o master.zip
cd caffe-master
mkdir build
cd build

# Prepare Python binding (pycaffe)
cd python
for req in $(cat requirements.txt); do sudo pip install $req; done
echo "export PYTHONPATH=$(pwd):$PYTHONPATH " >> ~/.bash_profile # to be able to call "import caffe" from Python
source ~/.bash_profile # Update shell 
cd ..

# Compile caffe and pycaffe
cd ..
cp Makefile.config.example Makefile.config
sed -i '8s/.*/CPU_ONLY := 1/' Makefile.config # CPU only
cd build
cmake ..
sudo make
sudo make all -j4 # 4 is the number of parallel threads for compilation
#make pycaffe
sudo make test
sudo make runtest
#make matcaffe
sudo make distribute
cd ..

# Franck bonus for other work
sudo pip install pydot
sudo apt-get install -y graphviz