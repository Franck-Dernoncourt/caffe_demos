# To install caffe and pycaffe on Ubuntu 14.04 x64 (also tested on Kubunty 14.10 x64). CPU only
# Usage: Execute "./compile_caffe_ubuntu_14.04.sh", wait for it to finish (~30 to 60 minutes), then open a new shell.


#http://caffe.berkeleyvision.org/install_apt.html : (general install info: http://caffe.berkeleyvision.org/installation.html)
sudo apt-get update
#sudo apt-get upgrade -y # If you are OK getting prompted
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y -q -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" # If you are OK with all defaults

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
sudo apt-get install -y python-scipy # in case pip failed
sudo apt-get install -y python-nose
#sudo chmod 777 /usr/local/man/man1/ # http://stackoverflow.com/questions/22753738/pip-install-matplotlib-error-error-usr-local-man-man1-nosetests-1-permission 
sudo pip install scikit-image # to fix https://github.com/BVLC/caffe/issues/50


# Get caffe (http://caffe.berkeleyvision.org/installation.html#compilation)
cd
mkdir caffe
cd caffe
wget https://github.com/BVLC/caffe/archive/master.zip
unzip -o master.zip
cd caffe-master

# Prepare Python binding (pycaffe)
cd python
for req in $(cat requirements.txt); do sudo pip install $req; done
echo "export PYTHONPATH=$(pwd):$PYTHONPATH " >> ~/.bash_profile # to be able to call "import caffe" from Python after reboot
source ~/.bash_profile # Update shell 
cd ..

# Compile caffe and pycaffe
cp Makefile.config.example Makefile.config
sed -i '8s/.*/CPU_ONLY := 1/' Makefile.config # CPU only
mkdir build
cd build
cmake ..
cd ..
sudo make
sudo make all -j4 # 4 is the number of parallel threads for compilation 
sudo make pycaffe
sudo make test
sudo make runtest
#make matcaffe
sudo make distribute

# Franck bonus for other work
sudo pip install pydot
sudo apt-get install -y graphviz
sudo pip install scikit-learn

# At the end, you need to run "source ~/.bash_profile" manually or start a new shell to be able to do 'python import caffe', 
# because one cannot source in a bash script. (http://stackoverflow.com/questions/16011245/source-files-in-a-bash-script)

# TODO: define number of CPUs + ROOTDIR for caffe