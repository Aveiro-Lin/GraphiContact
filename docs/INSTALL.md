## Installation

Our codebase is developed based on Ubuntu 16.04 and NVIDIA GPU cards. 

### Requirements
- Python 3.7
- Pytorch 1.13.1
- torchvision 0.5.0
- cuda 10.1

### Setup with Conda

We suggest to create a new conda environment and install all the relevant dependencies. 

```bash
# Create a new environment
conda create --name gphicon python=3.7
conda activate gphicon

# Install Pytorch
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch

export INSTALL_DIR=$PWD

# Install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

# Install OpenDR
pip install matplotlib
pip install git+https://gitlab.eecs.umich.edu/ngv-python-modules/opendr.git

# Install GraphiContact
cd $INSTALL_DIR
git clone --recursive https://anonymous.4open.science/r/GraphiContact.git
cd GraphiContact
python setup.py build develop

# Install requirements
pip install -r requirements.txt

# Install manopth
cd $INSTALL_DIR
cd GraphiContact
pip install ./manopth/.

unset INSTALL_DIR
```


