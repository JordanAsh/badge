import numpy as np
import pdb
from .strategy import Strategy
from sklearn.neighbors import NearestNeighbors
import pickle
from datetime import datetime
from sklearn.metrics import pairwise_distances

class CoreSet(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, tor=1e-4):
        super(CoreSet, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.tor = tor

    def furthest_first(self, X, X_set, n):
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []

        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        return idxs

    def query(self, n):
        t_start = datetime.now()
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        lb_flag = self.idxs_lb.copy()
        embedding = self.get_embedding(self.X, self.Y)
        embedding = embedding.numpy()

        chosen = self.furthest_first(embedding[idxs_unlabeled, :], embedding[lb_flag, :], n)

        return idxs_unlabeled[chosen]


    def query_old(self, n):
        lb_flag = self.idxs_lb.copy()
        embedding = self.get_embedding(self.X, self.Y)
        embedding = embedding.numpy()

        print('calculate distance matrix')
        t_start = datetime.now()
        dist_mat = np.matmul(embedding, embedding.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(len(self.X), 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        dist_mat = np.sqrt(dist_mat)
        print(datetime.now() - t_start)
        print('calculate greedy solution')
        t_start = datetime.now()
        mat = dist_mat[~lb_flag, :][:, lb_flag]

        for i in range(n):
            if i % 10 == 0:
                print('greedy solution {}/{}'.format(i, n))
            mat_min = mat.min(axis=1)
            q_idx_ = mat_min.argmax()
            q_idx = np.arange(self.n_pool)[~lb_flag][q_idx_]
            lb_flag[q_idx] = True
            mat = np.delete(mat, q_idx_, 0)
            mat = np.append(mat, dist_mat[~lb_flag, q_idx][:, None], axis=1)

        print(datetime.now() - t_start)
        opt = mat.min(axis=1).max()

        bound_u = opt
        bound_l = opt/2.0
        delta = opt

        xx, yy = np.where(dist_mat <= opt)
        dd = dist_mat[xx, yy]

        lb_flag_ = self.idxs_lb.copy()
        subset = np.where(lb_flag_==True)[0].tolist()

        SEED = 5
        sols = None

        if sols is None:
            q_idxs = lb_flag
        else:
            lb_flag_[sols] = True
            q_idxs = lb_flag_
        print('sum q_idxs = {}'.format(q_idxs.sum()))

        return np.arange(self.n_pool)[(self.idxs_lb ^ q_idxs)]
