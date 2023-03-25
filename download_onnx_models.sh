#!/bin/bash

apt-get install git-lfs
git lfs install

git clone -b onnx https://huggingface.co/runwayml/stable-diffusion-v1-5
