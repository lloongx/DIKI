#!bin/bash

cuda_version=$1

# create enviroment using Miniconda (or Anaconda)
conda create -n diki python=3.11.4
conda activate diki

# install pytorch
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=$cuda_version -c pytorch -c nvidia

# install other dependencies
pip install -r requirements.txt
