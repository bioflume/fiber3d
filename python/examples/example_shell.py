# Standard imports
from __future__ import division, print_function
import numpy as np
import scipy.linalg as scla
import scipy.sparse as scsp
import scipy.sparse.linalg as scspla
import time
import sys
from scipy.spatial import ConvexHull
import george
from george import HODLRSolver
sys.path.append('../')

# Local imports
from body import body
from shape_gallery import shape_gallery
from quadratures import Smooth_Closed_Surface_Quadrature_RBF
from periphery import periphery
from kernels import kernels
from quaternion import quaternion

def low_rank_approx(SVD=None, A=None, r=1):
  if SVD is None:
    SVD = np.linalg.svd(A, full_matrices=False)
  u, s, v = SVD
  Ar = np.zeros((len(u), len(v)))
  for i in xrange(r):
    Ar += s[i] * np.outer(u.T[i], v[i])
  return Ar

def check_symmetric(a, rtol=1e-05, atol=1e-08):
  return np.allclose(a, a.T, rtol=rtol, atol=atol)

if __name__ == '__main__':

  # Build the shell
  # ellipsoid
  per_a = 15
  per_b = 25
  per_c = 25
  Nper = 400
  nodes, normals, h, gradh = shape_gallery.shape_gallery('ellipsoid',
                                                        Nper,
                                                        a = per_a,
                                                        b = per_b,
                                                        c = per_c)
  normals = -normals
  hull = ConvexHull(nodes)
  triangles = hull.simplices

  # quadrature
  quadrature_weights = Smooth_Closed_Surface_Quadrature_RBF.Smooth_Closed_Surface_Quadrature_RBF(nodes,
                                                                                                triangles,
                                                                                                h,
                                                                                                gradh)
  # Build shell class
  shell = periphery.Periphery(np.array([0., 0., 0.]), quaternion.Quaternion([1.0, 0.0, 0.0, 0.0]), nodes, normals, quadrature_weights)
  # Compute singularity subtraction vectors
  shell.get_singularity_subtraction_vectors(eta = 1)

  # Precompute shell's r_vectors and normals
  trg_shell_surf = shell.get_r_vectors()
  normals_shell = shell.get_normals()                                                                                            
  weights = shell.quadrature_weights
  shell_stresslet = kernels.stresslet_kernel_times_normal_numba(trg_shell_surf, normals_shell, eta = 1)
  N = shell.Nblobs
  I = np.zeros((3*N, 3*N))
  for i in range(N):
    I[3*i:3*(i+1), 3*i+0] = shell.ex[3*i:3*(i+1)] / weights[i]
    I[3*i:3*(i+1), 3*i+1] = shell.ey[3*i:3*(i+1)] / weights[i]
    I[3*i:3*(i+1), 3*i+2] = shell.ez[3*i:3*(i+1)] / weights[i]
  I_vec = np.ones(N*3)
  I_vec[0::3] /= (1.0 * weights)
  I_vec[1::3] /= (1.0 * weights)
  I_vec[2::3] /= (1.0 * weights)
  shell_stresslet += -I - np.diag(I_vec)
  # Similarly, save shell's complementary matrix
  shell_complementary = kernels.complementary_kernel(trg_shell_surf, normals_shell)
  M = shell_stresslet + shell_complementary
  

  # Shape of the matrix:
  row, col = M.shape
  print('Shape of the matrix: ', row, col)

  # SVD of the matrix
  #svals = scla.svdvals(M)
  #print('First 10 singular values: ', svals[:10])
  #print('Last 10 singular values: ', svals[-10:])

  # Eigenvalue decomposition of the matrix
  #evals = np.linalg.eigvals(M)
  #print('Is all eigenvalues are positive: ', np.all(evals > 0))
  
  # Symmetric
  #print('Is the matrix M symmetric? : ', check_symmetric(M))


  a_LU = time.time()
  (LU, P) = scla.lu_factor(M)
  t_LU = time.time() - a_LU
  print('LU factorization of M took ', t_LU, ' secs')
