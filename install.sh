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

if [ ! -f "checkpoints/depth_pro.pt" ]; then
  source get_pretrained_models.sh
fi

cd ..

# Install marigold

if [ ! -d "Marigold" ]; then
    git clone https://github.com/prs-eth/Marigold.git
fi

cd Marigold

pip install -r requirements.txt
# pip install -e .

cd ../..

# install the other requirements
pip install -r requirements.txt

