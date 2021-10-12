#!/usr/bin/env bash
set -e

wget https://www.dropbox.com/s/i5tnuael9fogw16/spec-github-data.zip
unzip spec-github-data.zip
mkdir data/dataset_folders
rm spec-github-data.zip

mkdir -p $HOME/.torch/models/
mv data/yolov3.weights $HOME/.torch/models/