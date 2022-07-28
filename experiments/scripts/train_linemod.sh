#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 /home/wuyi/wym/EFN6D/Object-RPE-master/EFN6D/tools/train.py
