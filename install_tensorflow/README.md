## Setting up a Tensorflow from Scratch (Software)
A detailed guide to setting up your machine for deep learning research. Includes instructions to install drivers, tools and various deep learning frameworks. 
This was tested on the following machine:
* 64 bit machine with Nvidia GTX 745
* Ubuntu 16.04
* python 2.7.12
* CUDA 7.5
* cuDNN 5.0
* tensorflow 0.11
 

There are several great guides with a similar goal. Some are limited in scope, while others are not up to date. This guide is based on (with some portions copied verbatim from):
* [Setting up a Deep Learning Machine from Scratch](https://github.com/saiprashanths/dl-setup/blob/master/README.md)
* [Caffe Installation for Ubuntu](https://github.com/tiangolo/caffe/blob/ubuntu-tutorial-b/docs/install_apt2.md)
* [Running a Deep Learning Dream Machine](http://graphific.github.io/posts/running-a-deep-learning-dream-machine/)

### Table of Contents
* [Basics](#basics)
* [Nvidia Drivers](#nvidia-drivers)
* [CUDA](#cuda)
* [cuDNN](#cudnn)
* [Python Packages](#python-packages)
* [Tensorflow](#tensorflow)
* [Ipython notebook](#Ipython-notebook)

### Basics
* First, open a terminal and run the following commands to make sure your OS is up-to-date

        sudo apt update  
        sudo apt upgrade  
        sudo apt install build-essential cmake g++ gfortran git pkg-config python-dev software-properties-common wget
        sudo apt-get autoremove 
        sudo rm -rf /var/lib/apt/lists/*

### Nvidia Drivers
* Find your graphics card model

        lspci | grep -i nvidia

* We will install the drivers using apt-get. Check if your latest driver exists in the ["Proprietary GPU Drivers" PPA](https://launchpad.net/~graphics-drivers/+archive/ubuntu/ppa). Note that the latest drivers are necessarily the most stable. It is advisable to install the driver version recommended on that page. Add the "Proprietary GPU Drivers" PPA repository.
        sudo add-apt-repository ppa:graphics-drivers/ppa
        sudo apt update
        
* We will choose the approriate driver from "Software and Updates". Go to "Additional Drivers" and select the GPU Driver. At the time (Sept.30 2016) of this writing, the recommended version is 361.45.18. Select and install the driver.

* Restart your system

        sudo shutdown -r now
        
* Check to ensure that the correct version of NVIDIA drivers are installed

        cat /proc/driver/nvidia/version
        
### CUDA
* CUDA can be installed properly using apt-get.
        sudo apt install nvidia-cuda-dev
        sudo apt install nvidia-cuda-toolkit

        
* Check to ensure the correct version of CUDA is installed

        nvcc -V
        
* Restart your computer

        sudo shutdown -r now
        
#### Checking your CUDA Installation (Optional)
* Install the samples in the CUDA directory. Compile them (takes a few minutes):

        /usr/local/cuda/bin/cuda-install-samples-7.5.sh ~/cuda-samples
        cd ~/cuda-samples/NVIDIA*Samples
        make -j $(($(nproc) + 1))
        
**Note**: (`-j $(($(nproc) + 1))`) executes the make command in parallel using the number of cores in your machine, so the compilation is faster

* Run deviceQuery and ensure that it detects your graphics card and the tests pass

        bin/x86_64/linux/release/deviceQuery
        
### cuDNN
* cuDNN is a GPU accelerated library for DNNs. It can help speed up execution in many cases. To be able to download the cuDNN library, you need to register in the Nvidia website at [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn). This can take anywhere between a few hours to a couple of working days to get approved. Once your registration is approved, download **cuDNN v5.1 for Linux**. 

* Extract and copy the files

        cd ~/Downloads/
        tar -xvf cudnn-8.0-linux-x64-v5.1.tgz 
        cd cuda
        sudo cp -P include/cudnn.h /usr/include/
        sudo cp -P lib64/libcudnn* /usr/lib/x86_64-linux-gnu/
        sudo chmod a+r /usr/lib/x86_64-linux-gnu/libcudnn*
        
### Check
* You can do a check to ensure everything is good so far using the `nvidia-smi` command. This should output some stats about your GPU

### Python Packages
* Install some useful Python packages using apt-get. There are some version incompatibilities with using pip install and TensorFlow ( see https://github.com/tensorflow/tensorflow/issues/2034)
 
        sudo apt-get update && apt-get install -y python-numpy python-scipy python-nose \
                                                python-h5py python-skimage python-matplotlib \
		                                python-pandas python-sklearn python-sympy
        sudo apt-get clean && sudo apt-get autoremove
        rm -rf /var/lib/apt/lists/*
 

### Tensorflow
* This installs v0.11 with GPU support. Instructions below are from [here](https://www.tensorflow.org/get_started/os_setup.html). Follow the instructions to get the latest support. For the v0.11 version through pip install, as used here. 

        sudo apt-get install python-pip python-dev
        # Ubuntu/Linux 64-bit, GPU enabled, Python 2.7
        # Requires CUDA toolkit 7.5 and CuDNN v5.
        export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0rc0-cp27-none-linux_x86_64.whl
        sudo pip install --upgrade $TF_BINARY_URL
        

* Run a test to ensure your Tensorflow installation is successful. When you execute the `import` command, there should be no warning/error.

        python
        >>> import tensorflow as tf
        >>> exit()
        python -c 'import os; import inspect; import tensorflow; print(os.path.dirname(inspect.getfile(tensorflow)))'
        python -m tensorflow.models.image.mnist.convolutional

### Ipython notebook
* Ipython and notebook is installed through pip.
        sudo pip install pip --upgrade
        sudo pip install ipython --upgrade
        sudo pip install ipython[all] --upgrade