#!/usr/bin/env bash

# Install SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# Install ml-depth-pro
set -e

mkdir -p dependencies

cd dependencies

if [ ! -d "ml-depth-pro" ]; then
    git clone https://github.com/apple/ml-depth-pro.git
fi

cd ml-depth-pro

pip install -e .

source get_pretrained_models.sh

cd ../..

# install the other requirements
pip install -r requirements.txt

