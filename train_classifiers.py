#!/usr/bin/env python

"""Python functions that train the classifiers descrbied in
model/classifiers.py using either MNIST or fMNIST"""

# Standard libraries
import argparse
import numpy as np
import os
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import datetime

# User-made scripts
from models import classifiers

from src.load_mnist import *
from src.mnist_reader import *


def main():

    print(args)

    # (hyper)parameters
    model_name = args.model
    dataset = args.dataset
    class_use = np.array(args.class_use)
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    momentum = args.momentum
    c_dim = 1
    img_size = 28

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class_use_str = "".join(map(str, class_use))
    y_dim = class_use.shape[0]
    newClass = range(0, y_dim)
    test_size = 100

    save_folder_root = './models/classifiers/'
    save_folder = os.path.join(save_folder_root,
                               f"{model_name}_{dataset}_{class_use_str}_classifier")
    summary_writer = SummaryWriter(save_folder + "/checkpoints")

    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(save_folder + "/checkpoints", exist_ok=True)

    # Dataset preparation
    if dataset == 'mnist':
        trX, trY, tridx = load_mnist_classSelect('train', class_use, newClass)
        vaX, vaY, vaidx = load_mnist_classSelect('val', class_use, newClass)
        teX, teY, teidx = load_mnist_classSelect('test', class_use, newClass)
    elif dataset == 'fmnist':
        trX, trY, tridx = load_fashion_mnist_classSelect('train', class_use, newClass)
        vaX, vaY, vaidx = load_fashion_mnist_classSelect('val', class_use, newClass)
        teX, teY, teidx = load_fashion_mnist_classSelect('test', class_use, newClass)
    else:
        print('dataset must be ''mnist'' or ''fmnist''!')

    # Number of batches per epoch
    batch_idxs = len(trX) // batch_size
    batch_idxs_val = len(vaX) // test_size

    # Training
    ce_loss = nn.CrossEntropyLoss()

    classifier = classifiers.InceptionNetDerivative(num_classes=y_dim).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=momentum)

    print(classifier)

    for epoch in range(epochs):
        for idx in range(batch_idxs):
            # Gather input and labels for current batch
            labels = torch.from_numpy(trY[idx * batch_size:(idx + 1) * batch_size]).long().to(device)
            inp = trX[idx * batch_size:(idx + 1) * batch_size]
            inp = torch.from_numpy(inp)
            inp = inp.permute(0, 3, 1, 2).float()
            inp = inp.to(device)

            # Calculate loss and acc, then step with optimizer
            optimizer.zero_grad()
            probs = classifier(inp)
            loss = ce_loss(probs, labels)
            loss.backward()

            acc = (probs.argmax(dim=-1) == labels).float().mean()
            optimizer.step()

            print("[{}] Train Epoch {:03d}/{:03d}, Batch Size = {}, \
                  loss = {:.3f}, acc = {}".format(
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), epoch,
                    epochs, batch_size, loss.item(), acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="inception",
                        help="Specification of model to be trained.")
    parser.add_argument("--dataset", type=str, default="mnist",
                        help="Specification of dataset to be used.")
    parser.add_argument("--class_use", type=int, default=[3, 8],
                        help="Specification of what classes to use" +
                             "To specify multiple, use \" \" to" +
                             "separate them. Example \"3 8\".")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Specification of batch size to be used.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Specification of training epochs.")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Specification of learning rate for SGD")
    parser.add_argument("--momentum", type=float, default=0.5,
                        help="Specification of Momentum for SGD")
    parser.add_argument("--seed", type=int, default=1,
                        help="Specification of random seed of this run")

    args = parser.parse_args()

    main()
