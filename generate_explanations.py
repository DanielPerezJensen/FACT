"""
Generates explanation of given classifier/gce combinations,
much like in figure 3 of the original paper.
"""

# import standard libraries
import argparse
import numpy as np
import scipy.io as sio
import os
import torch

# Import user defined librariers
from models import classifiers
from src.models.CVAE import Decoder, Encoder

import src.util as util
import src.plotting as plotting
from src.GCE import GenerativeCausalExplainer
from src.load_mnist import *


def main():

    model_name = args.model_file.split("_")[0]
    data = args.model_file.split("_")[1]
    data_classes = np.array(list(args.model_file.split("_")[2]), dtype=int)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ylabels = range(0, len(data_classes))

    # Gather K, L and image size from dataset
    if data.lower() == "mnist":
        X, Y, tridx = load_mnist_classSelect('train', data_classes, ylabels)
        vaX, vaY, vaidx = load_mnist_classSelect('val', data_classes, ylabels)
        K, L = 1, 7
    elif data.lower() == "fmnist":
        X, Y, tridx = load_fashion_mnist_classSelect('train', data_classes, ylabels)
        vaX, vaY, vaidx = load_fashion_mnist_classSelect('val', data_classes, ylabels)
        K, L = 2, 4

    ntrain, nrow, ncol, c_dim = X.shape
    x_dim = nrow * ncol
    y_dim = len(data_classes)

    # Import stated model
    if model_name.lower() == "inceptionnet":
        classifier = classifiers.InceptionNetDerivative(num_classes=y_dim).to(device)
    elif model_name.lower() == "resnet":
        classifier = classifiers.ResNetDerivative(num_classes=y_dim).to(device)
    elif model_name.lower() == "densenet":
        classifier = classifiers.DenseNetDerivative(num_classes=y_dim).to(device)

    encoder = Encoder(K+L, c_dim, x_dim).to(device)
    decoder = Decoder(K+L, c_dim, x_dim).to(device)

    # Load GCE from stored model
    gce = GenerativeCausalExplainer(classifier, decoder, encoder, device)

    # Load trained weights
    checkpoint = torch.load(f"models/GCEs/{args.model_file}_gce/model.pt")

    gce.classifier.load_state_dict(checkpoint["model_state_dict_classifier"])
    gce.encoder.load_state_dict(checkpoint["model_state_dict_encoder"])
    gce.decoder.load_state_dict(checkpoint["model_state_dict_decoder"])

    # Perform dummy train to set proper parameters
    traininfo = gce.train(X, K, L,
                          steps=1,
                          Nalpha=15,
                          Nbeta=50,
                          lam=0,
                          batch_size=64,
                          lr=0)

    # --- compute final information flow ---
    I = gce.informationFlow()
    Is = gce.informationFlow_singledim(range(0, K+L))
    print('Information flow of K=%d causal factors on classifier output:' % K)
    print(Is[:K])
    print('Information flow of L=%d noncausal factors on classifier output:' % L)
    print(Is[K:])

    # --- generate explanation and create figure ---
    sample_ind = np.concatenate((np.where(vaY == 0)[0][:4],
                                 np.where(vaY == 1)[0][:4]))
    x = torch.from_numpy(vaX[sample_ind])
    zs_sweep = [-3., -2., -1., 0., 1., 2., 3.]
    Xhats, yhats = gce.explain(x, zs_sweep)

    os.makedirs(f"reports/figures/GCEs/{model_name + '_' + data}", exist_ok=True)
    plotting.plotExplanation(1.-Xhats, yhats, save_path=f'reports/figures/GCEs/{model_name + "_" + data}/{model_name}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_file", type=str, default="inceptionnet_mnist_38",
                        help="Specification of what model we are using.")

    args = parser.parse_args()

    main()
