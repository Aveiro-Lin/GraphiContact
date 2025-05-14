## Installation

Our codebase is developed based on Ubuntu 20.04 and NVIDIA GPU cards. 

### Requirements
- Python 3.9
- Pytorch 1.10.0
- torchvision 0.11.0
- cuda 11.3

### Setup with Conda

We suggest to create a new conda environment and install all the relevant dependencies. 

```bash
# Create a new environment
conda create --name graphi_con python=3.9
conda activate graphi_con

# Install Pytorch
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch -c conda-forge
# If running `python -c "import torch"` results in the error `undefined symbol: iJIT_NotifyEvent`, then execute the following command.
conda install mkl==2021.4.0 -c conda-forge

# Install cuda
conda install cuda-nvcc==11.3.58 -c nvidia
conda install cuda==11.3 -c nvidia

export INSTALL_DIR=$PWD

# Install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82
unset CUDA_HOME
conda install packaging "numpy<2" -c conda-forge
python setup.py install --cuda_ext --cpp_ext

# Install pytorch3d
conda install -c iopath -c conda-forge iopath
conda install -c bottler nvidiacub
conda install -c pytorch3d -c conda-forge pytorch3d

# Install GraphiContact
cd $INSTALL_DIR
git clone --recursive https://github.com/Aveiro-Lin/GraphiContact.git
cd GraphiContact
python setup.py build develop
pip install -r requirements.txt
conda install monai==1.0.1 bitsandbytes -c conda-forge
pip install smplx==0.1.28 --no-deps
pip install pyopengl==3.1.4 --upgrade

# Install OpenDR from GraphiContact
cd $INSTALL_DIR
cd GraphiContact/opendr
cd opendr/contexts/
wget http://files.is.tue.mpg.de/mloper/opendr/osmesa/OSMesa.Linux.x86_64.zip
cd -
python setup.py build && python setup.py install
export OPENDR_PATH=$(python -c "import opendr; print(opendr.__path__[0])")
cp $OPENDR_PATH/contexts/_constants.py $CONDA_PREFIX/lib/python3.9/site-packages/

# Install manopth from GraphiContact
cd $INSTALL_DIR
cd GraphiContact/manopth
pip install .

unset INSTALL_DIR
```