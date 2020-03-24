## Use this script to perform quick tests on the codebase for verification

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from eucl_dist.cpu_dist import dist
from eucl_dist.gpu_dist import dist as gdist

a = np.random.rand(10000,3)
b = np.random.rand(10000,3)
print(np.allclose(pairwise_distances(a,b, 'sqeuclidean'), dist(a,b), atol=1e-5))
print(np.allclose(pairwise_distances(a,b, 'sqeuclidean'), gdist(a,b), atol=1e-5))

a = np.random.rand(800,2048).astype(np.float32)
b = np.random.rand(800,2048).astype(np.float32)
print(np.allclose(pairwise_distances(a,b, 'sqeuclidean'), dist(a,b), atol=1e-5))
print(np.allclose(pairwise_distances(a,b, 'sqeuclidean'), gdist(a,b), atol=1e-5))
