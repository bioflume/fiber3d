from __future__ import division, print_function
import numpy as np
from kernels import kernels
try:
  from numba import njit, prange
except ImportError:
  print('Numba not found')
  pass


def single_layer_double_layer_numba(r_vectors,
                                    normals,
                                    density,
                                    quadrature_weights,
                                    singularity_substraction_matrix,
                                    ex, ey, ez,
                                    eta=1.0,
                                    epsilon_distance=1e-10,
                                    *args, **kwargs):
  '''
  Compute the mobility like a combination of single and double layer potentials
  
  M^pq = delta_pq/2 * I + S^pq + D^pq

  where S^qp and D^pq are the single and double
  layer kernels between points p and q.
  '''

  u_single = kernels.oseen_kernel_times_density_numba(r_vectors, density, eta, epsilon_distance)
  u_double = kernels.stresslet_kernel_times_normal_times_density_numba(r_vectors, normals, density, eta, epsilon_distance)
  u_traction = kernels.traction_kernel_times_normal_times_density_numba(r_vectors, normals, density, eta, epsilon_distance)

  density = np.reshape(density, (density.size // 3 , 3))
  
  if False:
    w = np.flipud(quadrature_weights) / quadrature_weights
    wd = (density * w[:,None]).flatten()
    correction = np.dot(wd, singularity_substraction_matrix)
    # print(w)
  elif True:
    cx = (density / quadrature_weights[:,None]).flatten() * ex.flatten() 
    #cy = (density / quadrature_weights[:,None]).flatten() * ey.flatten()
    #cz = (density / quadrature_weights[:,None]).flatten() * ez.flatten()

 
  # return 0.5 * (density / quadrature_weights[:,None]).flatten() + u_single + u_double
  # return 1.0 * (density / quadrature_weights[:,None]).flatten() + u_single + u_double - correction
  # return 0.5 * (density / quadrature_weights[:,None]).flatten() + u_single
  return (0.5 / eta) * (density / quadrature_weights[:,None]).flatten() + u_single + u_double
  # return (0.5 / eta) * (density / quadrature_weights[:,None]).flatten() + u_single + u_traction
  # return -(1.0 / eta) * (density / quadrature_weights[:,None]).flatten() + u_single + u_double + cx


