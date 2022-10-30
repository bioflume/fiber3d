from __future__ import division, print_function
import numpy as np
import sys
sys.path.append('../')
sys.path.append('../../../RigidMultiblobsWall/')

from kernels import kernels
from mobility import mobility as mob
from utils import timer


if __name__ == '__main__':

  # np.random.seed(1)  
  N = 100
  r = np.random.rand(N, 3)
  test = 'oseen_product'
  # test = 'stresslet_normal_density'

  if test == 'oseen':
    timer.timer('Oseen')
    G = kernels.oseen_tensor(r)
    timer.timer('Oseen')
    timer.timer('RPY')
    RPY = mob.rotne_prager_tensor(r, 1.0, 1e-10)
    timer.timer('RPY')
    np.fill_diagonal(RPY, 0.0)
    
    error = np.linalg.norm(G - RPY) / np.linalg.norm(G)
    print('|G - RPY| / |G| = ', error)

  elif test == 'oseen_product':
    density = np.random.rand(N, 3)

    timer.timer('Numpy')
    G = kernels.oseen_tensor(r)
    u = np.dot(G, density.flatten())
    timer.timer('Numpy')
    u_numba = kernels.oseen_kernel_times_density_numba(r, density)
    timer.timer('Numba')
    u_numba = kernels.oseen_kernel_times_density_numba(r, density)
    timer.timer('Numba')

    u_pycuda = kernels.oseen_kernel_source_target_pycuda(r, r, density)
    timer.timer('pycuda')
    u_pycuda = kernels.oseen_kernel_source_target_pycuda(r, r, density)
    timer.timer('pycuda')

    
    error_numba = np.linalg.norm(u - u_numba) / np.linalg.norm(u)
    error_pycuda = np.linalg.norm(u - u_pycuda) / np.linalg.norm(u)

    print('|u - u_numba| / |u| = ', error_numba)
    print('|u - u_pycuda| / |u| = ', error_pycuda)
    # print('u = \n', u)
    # print('u_numba = \n', u_numba)
    # print('u_pycuda = \n', u_pycuda)
    
  elif test == 'stresslet_normal':
    normal = np.random.rand(N, 3)
    
    # Numpy
    timer.timer('Stresslet_numpy')
    Snormal = kernels.stresslet_kernel_times_normal(r, normal)
    timer.timer('Stresslet_numpy')

    # Numba
    Snormal_numba = kernels.stresslet_kernel_times_normal_numba(r, normal)
    timer.timer('Stresslet_numba')
    Snormal_numba = kernels.stresslet_kernel_times_normal_numba(r, normal)
    timer.timer('Stresslet_numba')

    # Difference
    difference = Snormal - Snormal_numba
    error = np.linalg.norm(difference) / np.linalg.norm(Snormal)
    print('|Snormal - Snormal_numba| / |Snormal| = ', error)
    # print('Svev_numba = \n', Snormal_numba)
    # print('Svev = \n', Snormal)

  elif test == 'stresslet_normal_density':
    normal = np.random.rand(N, 3)
    density = np.random.rand(N, 3)

    # Numpy
    timer.timer('Stresslet_numpy')
    Snormal = kernels.stresslet_kernel_times_normal(r, normal)
    u = np.dot(Snormal, density.flatten())
    timer.timer('Stresslet_numpy')

    # Numba
    u_numba = kernels.stresslet_kernel_times_normal_times_density_numba(r, normal, density)
    timer.timer('Stresslet_numba')
    u_numba = kernels.stresslet_kernel_times_normal_times_density_numba(r, normal, density)
    timer.timer('Stresslet_numba')

    # Difference
    difference = u - u_numba
    error = np.linalg.norm(difference) / np.linalg.norm(u)
    print('|u - u_numba| / |u| = ', error)
    # print(u)
    # print(u_numba)    

  elif test == 'complementary':
    normal = np.random.rand(N, 3)
    density = np.random.rand(N, 3)

    # Numba, matrix free
    u = kernels.complementary_kernel_times_density_numba(r, normal, density)
    timer.timer('numba_matrix_free')
    u = kernels.complementary_kernel_times_density_numba(r, normal, density)
    timer.timer('numba_matrix_free')
    
    # Numba, full matrix
    Nk = kernels.complementary_kernel(r, normal)
    timer.timer('python')
    Nk = kernels.complementary_kernel(r, normal)
    u_full = np.dot(Nk, density.flatten())
    timer.timer('python')

    # Difference
    difference = u - u_full
    error = np.linalg.norm(difference) / np.linalg.norm(u)
    print('|u - u_full| / |u| = ', error)
    # print('difference = \n', difference)
  

  timer.timer('', print_all=True)
