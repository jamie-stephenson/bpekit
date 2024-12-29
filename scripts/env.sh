#!/bin/bash
# Script to set up local enivronment on each node 

mount_dir="$1"

# TODO: automatically create `slurmlogs` in the correct place
mkdir -p $mount_dir/slurmlogs

#--------RUST---------
sudo NEEDRESTART_MODE=l apt-get -o DPkg::Lock::Timeout=60 install -y libclang-dev
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
#---------------------

#------OPEN MPI-------
sudo NEEDRESTART_MODE=l apt-get -o DPkg::Lock::Timeout=60 install -y openmpi-bin openmpi-common libopenmpi-dev 
#---------------------

#-PYTHON ENVIRONMENT--
# If you want a specific python version you can use deadsnakes:
sudo NEEDRESTART_MODE=l apt-get -o DPkg::Lock::Timeout=60 -y install python3.12-venv
python3 -m venv ~/envs/bpekit
source ~/envs/bpekit/bin/activate
pip install maturin[patchelf]
deactivate
#---------------------     
