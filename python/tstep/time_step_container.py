from __future__ import division, print_function
import numpy as np
from functools import partial
import scipy.linalg as scla
import scipy.sparse as scsp
import scipy.sparse.linalg as scspla
import sys
import time
import copy
import os

from scipy.spatial import ConvexHull

# OUR CLASSES
from tstep import fiber_matrices

import _pickle as cpickle

try:
  from numba import njit, prange
  from numba.typed import List
except ImportError:
  print('Numba not found')

class time_step_container(object):


  ##############################################################################################
  def __init__(self, prams, options, fibers, bodies, tstep):
    self.fibers = fibers
    self.bodies = bodies
    self.tstep = tstep

    self.fib_mats = []
    self.fib_mat_resolutions = np.array([])

    fib_mat = fiber_matrices.fiber_matrices(num_points = options.num_points, num_points_finite_diff = options.num_points_finite_diff, uprate_poten = options.uprate)
    fib_mat.compute_matrices()
    self.fib_mats.append(fib_mat)
    self.fib_mat_resolutions = np.append(self.fib_mat_resolutions, options.num_points)


    for k, fib in enumerate(self.fibers):
      if fib.num_points not in self.fib_mat_resolutions:
        # If fib_mat class for this resolution has not been created, then create and store
        fib_mat = fiber_matrices.fiber_matrices(num_points = fib.num_points, num_points_finite_diff = options.num_points_finite_diff, uprate_poten = options.uprate)
        fib_mat.compute_matrices()
        self.fib_mats.append(fib_mat)
        self.fib_mat_resolutions = np.append(self.fib_mat_resolutions, fib.num_points)


    return # init
  ##############################################################################################
