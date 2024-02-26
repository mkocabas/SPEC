#!/usr/bin/env bash
set -e

pip install gdown
gdown 1o4-ilb3GF7CqH0LWI6RE7fRWQ6qEyDzE
unzip spec-github-data.zip
mkdir data/dataset_folders
rm spec-github-data.zip

mkdir -p $HOME/.torch/models/
mv data/yolov3.weights $HOME/.torch/models/
