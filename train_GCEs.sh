#!/bin/bash

python train_GCE.py --model_file inceptionnet_mnist_38_classifier --train_steps 5000 --K 1 --L 7 --lam 0.05 --Nalpha 10 --Nbeta 50
python train_GCE.py --model_file densenet_mnist_38_classifier --train_steps 5000 --K 1 --L 7 --lam 0.05 --Nalpha 10 --Nbeta 50
python train_GCE.py --model_file resnet_mnist_38_classifier --train_steps 5000 --K 1 --L 7 --lam 0.05 --Nalpha 10 --Nbeta 50
python train_GCE.py --model_file inceptionnet_fmnist_034_classifier --train_steps 5000 --K 2 --L 4 --lam 0.05 --Nalpha 10 --Nbeta 50
python train_GCE.py --model_file densenet_fmnist_034_classifier --train_steps 5000 --K 2 --L 4 --lam 0.05 --Nalpha 10 --Nbeta 50
python train_GCE.py --model_file resnet_fmnist_034_classifier --train_steps 5000 --K 2 --L 4 --lam 0.05 --Nalpha 10 --Nbeta 50

# Note: Files found in models/GCEs were trained on a system with the following components:
# NVIDIA RTX 3060TI (using nvidia driver 460, and CUDA 11.2)
# AMD Ryzen 5 4600
# 32GB RAM