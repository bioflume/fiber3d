from __future__ import division, print_function
import numpy as np
import sys
sys.path.append('../')
from mobility import mobility as mob


if __name__ == '__main__':
  N = 100
  r = np.random.rand(N, 3)
  normals = np.random.rand(N, 3)
  density = np.random.rand(N, 3)
  weights = np.ones(N)

  u = mob.single_layer_double_layer_numba(r, normals, density, weights, 1.0)

  print('u = \n', u)

