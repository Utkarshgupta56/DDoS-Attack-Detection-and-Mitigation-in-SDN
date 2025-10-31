#!/bin/bash

# Clone and install Mininet first
git clone https://github.com/mininet/mininet.git
cd mininet
sudo ./util/install.sh -a
cd ..

# Add the deadsnakes PPA
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install hping3

# Install Python 3.7 and necessary packages
sudo apt install -y python3.7 python3.7-venv python3.7-dev python3.7-distutils

# Create virtual environment
python3.7 -m venv ryu37-env
source ryu37-env/bin/activate

# Clone Ryu repository
git clone https://github.com/faucetsdn/ryu.git

# Install Ryu dependencies
pip install -r ryu/tools/pip-requires

# Install Ryu
cd ryu
pip install .