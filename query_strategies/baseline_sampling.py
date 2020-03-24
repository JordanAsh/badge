import numpy as np
from torch.utils.data import DataLoader
from .strategy import Strategy
import pickle
from scipy.spatial.distance import cosine
import sys
import gc
from scipy.linalg import det
from scipy.linalg import pinv as inv
from copy import copy as copy
from copy import deepcopy as deepcopy
import torch
from torch import nn
import torchfile
from torch.autograd import Variable
import resnet
import vgg
import torch.optim as optim
import pdb
from torch.nn import functional as F
import argparse
import torch.nn as nn
from collections import OrderedDict
from scipy import stats
import time
import numpy as np
import scipy.sparse as sp
from itertools import product
import sklearn
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.utils.extmath import row_norms, squared_norm, stable_cumsum
from sklearn.utils.sparsefuncs_fast import assign_rows_csr
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils.validation import _num_samples
from sklearn.utils import check_array
from sklearn.utils import gen_batches
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.metrics.pairwise import rbf_kernel as rbf
from sklearn.externals.six import string_types
from sklearn.exceptions import ConvergenceWarning

def gram_red(L, L_inv, u_loc):
    n = np.shape(L_inv)[0]
    ms = np.array([False for i in range(n)])
    ms[u_loc] = True

    L_red = L[~ms][:, ~ms]

    D = L_inv[~ms][:, ~ms]
    e = L_inv[~ms][:, ms]
    f = L_inv[ms][:, ms]

    L_red_inv = D - e.dot(e.T) / f
    return L_red, L_red_inv

def gram_aug(L_Y, L_Y_inv, b_u, c_u):
    d_u = c_u - b_u.T.dot(L_Y_inv.dot(b_u))
    g_u = L_Y_inv.dot(b_u)

    L_aug = np.block([[L_Y, b_u],[b_u.T, c_u]])
    L_aug_inv = np.block([[L_Y_inv + g_u.dot(g_u.T/d_u), -g_u/d_u], [-g_u.T/d_u, 1.0/d_u]])

    return L_aug, L_aug_inv

def sample_k_imp(Phi, k, max_iter, rng=np.random):
    n = np.shape(Phi)[0]
    Ind = rng.choice(range(n), size=k, replace=False)

    if n == k:
        return Ind

    X = [False] * n
    for i in Ind:
        X[i] = True
    X = np.array(X)

    L_X = Phi[Ind, :].dot(Phi[Ind, :].T)

    L_X_inv = np.linalg.pinv(L_X)

    for i in range(1, max_iter):

        u = rng.choice(np.arange(n)[X])
        v = rng.choice(np.arange(n)[~X])

        for j in range(len(Ind)):
            if Ind[j] == u:
                u_loc = j

        L_Y, L_Y_inv = gram_red(L_X, L_X_inv, u_loc)

        Ind_red = [i for i in Ind if i != u]

        b_u = Phi[Ind_red, :].dot(Phi[[u], :].T)
        c_u = Phi[[u], :].dot(Phi[[u], :].T)
        b_v = Phi[Ind_red, :].dot(Phi[[v], :].T)
        c_v = Phi[[v], :].dot(Phi[[v], :].T)

        p = min(1, (c_v - b_v.T.dot(L_Y_inv.dot(b_v))) / (c_u - b_u.T.dot(L_Y_inv.dot(b_u))) )

        if rng.uniform() <= p:
            X[u] = False
            X[v] = True
            Ind = Ind_red + [v]
            L_X, L_X_inv = gram_aug(L_Y, L_Y_inv, b_v, c_v)

        if i % k == 0:
            print('Iter ', i)

    return Ind

class BaselineSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(BaselineSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        gradEmbedding = self.get_grad_embedding(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled]).numpy()
        chosen = sample_k_imp(gradEmbedding, n, max_iter= int(5 * n * np.log(n)))
        return idxs_unlabeled[chosen]
