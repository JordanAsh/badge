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
import numpy as np
import scipy.sparse as sp
from itertools import product
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
#from sklearn.externals.six import string_types
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances


def distance(X1, X2, mu):
    Y1, Y2 = mu
    X1_vec, X1_norm_square = X1
    X2_vec, X2_norm_square = X2
    Y1_vec, Y1_norm_square = Y1
    Y2_vec, Y2_norm_square = Y2
    dist = X1_norm_square * X2_norm_square + Y1_norm_square * Y2_norm_square - 2 * (X1_vec @ Y1_vec) * (X2_vec @ Y2_vec)
    # Numerical errors may cause the distance squared to be negative.
    assert np.min(dist) / np.max(dist) > -1e-4
    dist = np.sqrt(np.clip(dist, a_min=0, a_max=None))
    return dist


def init_centers(X1, X2, chosen, chosen_list,  mu, D2):
    if len(chosen) == 0:
        ind = np.argmax(X1[1] * X2[1])
        mu = [((X1[0][ind], X1[1][ind]), (X2[0][ind], X2[1][ind]))]
        D2 = distance(X1, X2, mu[0]).ravel().astype(float)
        D2[ind] = 0
    else:
        newD = distance(X1, X2, mu[-1]).ravel().astype(float)
        D2 = np.minimum(D2, newD)
        D2[chosen_list] = 0
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(Ddist)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in chosen: ind = customDist.rvs(size=1)[0]
        mu.append(((X1[0][ind], X1[1][ind]), (X2[0][ind], X2[1][ind])))
    chosen.add(ind)
    chosen_list.append(ind)
    print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
    return chosen, chosen_list, mu, D2


class BadgeSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(BadgeSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        embs, probs = self.get_embedding(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled], return_probs=True)
        embs = embs.numpy()
        probs = probs.numpy()

        # the logic below reflects a speedup proposed by Zhang et al.
        # see Appendix D of https://arxiv.org/abs/2306.09910 for more details
        m = (~self.idxs_lb).sum()
        mu = None
        D2 = None
        chosen = set()
        chosen_list = []
        emb_norms_square = np.sum(embs ** 2, axis=-1)
        max_inds = np.argmax(probs, axis=-1)

        probs = -1 * probs
        probs[np.arange(m), max_inds] += 1
        prob_norms_square = np.sum(probs ** 2, axis=-1)
        for _ in range(n):
            chosen, chosen_list, mu, D2 = init_centers((probs, prob_norms_square), (embs, emb_norms_square), chosen, chosen_list, mu, D2)
        return idxs_unlabeled[chosen_list]
