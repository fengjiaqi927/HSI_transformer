# Spectral Transformer with Dynamic Spatial Sampling and Gaussian Positional Embedding for Hyperspectral Image Classification
## Framework
![Alt text](./framework.png)

## Enveronment
conda create -n HSI python=3.7.11
conda activate HSI
pip install -r requirements.txt

## Attention
HSI cube is generated from superpixel region. It is time-consuming when generating in each batch. To speed up the dataload process, we generate the HSI cube before training network and all cube is storaged in SLIC_samples.npy.

## Usage
1. normal version
train:  'python train_IP_normal.py'
test:   'python test_IP_normal.py'

2. speed up version
train:  'python train_IP_speed_up.py'
test:   'python test_IP_speed_up.py'

## TODO
1. Add article link 
2. Upload code 
