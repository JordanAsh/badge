import numpy as np
from .strategy import Strategy
import pdb

class RandomSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(RandomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        inds = np.where(self.idxs_lb==0)[0]
        return inds[np.random.permutation(len(inds))][:n]
