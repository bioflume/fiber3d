from __future__ import division, print_function
import numpy as np
import sys
from functools import partial
sys.path.append('..')
from mpi4py import MPI
from numba import njit, prange

import stkfmm
import kernels
from utils import timer


@njit(parallel=True, fastmath=True)
def distance(r_vectors):
  x = np.copy(r_vectors[:,0])
  y = np.copy(r_vectors[:,1])
  z = np.copy(r_vectors[:,2])
  N = x.size
  distance2 = np.ones(N) * 1e+99
  for i in prange(N):
    for j in range(N):
      if i == j:
        continue
      xij = x[i] - x[j]
      yij = y[i] - y[j]
      zij = z[i] - z[j]
      d2 = xij*xij + yij*yij + zij*zij
      if d2 < distance2[i]:
        distance2[i] = d2
  d2_min = np.min(distance2)
  return np.sqrt(d2_min)



if __name__ == '__main__':
  print('# Start')
  # Set parameters
  nsrc_SL = 100000
  nsrc_DL = 100000
  ntrg = 100000
  phi = 1e-03
  L = np.power(4*np.pi * (nsrc_SL + nsrc_DL) / (3 * phi), 1.0/3.0) 
  eta = 1.0
  mult_order = 8
  max_pts = 1000
  N_max = 1e+07
  print('L = ', L)

  # Create sources and targets coordinates
  r_source_SL = np.random.rand(nsrc_SL, 3) * L
  r_source_DL = np.random.rand(nsrc_DL, 3) * L
  r_target = np.random.rand(ntrg, 3) * L

  # Test single layer potential
  if True:
    print('TESTING SINGLE LAYER POTENTIAL')
    # Init FMM
    timer.timer('FMM_oseen_init')
    pbc = stkfmm.PAXIS.NONE
    kernel = stkfmm.KERNEL.PVel
    kernels_index = stkfmm.KERNEL(kernel)
    fmm_PVel = stkfmm.STKFMM(mult_order, max_pts, pbc, kernels_index)
    kdimSL, kdimDL, kdimTrg = stkfmm.getKernelDimension(fmm_PVel, kernel)
    oseen_kernel_source_target_stkfmm_partial = partial(kernels.oseen_kernel_source_target_stkfmm, fmm_PVel=fmm_PVel)
    timer.timer('FMM_oseen_init')

    timer.timer('FMM_stresslet_init')
    pbc = stkfmm.PAXIS.NONE
    kernel = stkfmm.KERNEL.PVel
    kernels_index = stkfmm.KERNEL(kernel)
    fmmStress_PVel = stkfmm.STKFMM(mult_order, max_pts, pbc, kernels_index)
    kdimSL, kdimDL, kdimTrg = stkfmm.getKernelDimension(fmmStress_PVel, kernel)
    stresslet_kernel_source_target_stkfmm_partial = partial(kernels.stresslet_kernel_source_target_stkfmm, fmm_PVel=fmmStress_PVel)
    timer.timer('FMM_stresslet_init')

    print('\n\nKernels info: ')
    print('kdimSL  = ', kdimSL)
    print('kdimDL  = ', kdimDL)
    print('kdimTrg = ', kdimTrg, '\n\n')

    
    # Create source density
    density_SL = np.random.randn(nsrc_SL, 3) 
    
    # Call FMM first time
    timer.timer('FMM_oseen_call_tree')
    v_stkfmm_set_tree = oseen_kernel_source_target_stkfmm_partial(r_source_SL, r_target, density_SL)
    timer.timer('FMM_oseen_call_tree')
    
    # Call FMM second time
    timer.timer('FMM_oseen_call')
    v_stkfmm = oseen_kernel_source_target_stkfmm_partial(r_source_SL, r_target, density_SL)
    timer.timer('FMM_oseen_call')
    
    # Compute velocities with numba if system is not too large
    if np.sqrt((nsrc_SL) * ntrg) <= N_max:
      v_numba = kernels.oseen_kernel_source_target_numba(r_source_SL, r_target, density_SL)
      timer.timer('numba_oseen')
      v_numba = kernels.oseen_kernel_source_target_numba(r_source_SL, r_target, density_SL)
      timer.timer('numba_oseen')
      diff_0 = v_stkfmm_set_tree - v_numba
      diff_1 = v_stkfmm - v_numba
      
      # Compute errors
      print('\n\n')
      print('v_numba[0:6]          = ', v_numba[0:6])
      print('v_stkfmm[0:6]         = ', v_stkfmm[0:6])
      print('norm velocity         = ', np.linalg.norm(v_numba))
      print('relative L2 error (1) = ', np.linalg.norm(diff_0) / np.linalg.norm(v_numba))
      print('Linf error        (1) = ', np.linalg.norm(diff_0.flatten(), ord=np.inf))
      print('relative L2 error (2) = ', np.linalg.norm(diff_1) / np.linalg.norm(v_numba))
      print('Linf error        (2) = ', np.linalg.norm(diff_1.flatten(), ord=np.inf))
      print('\n\n\n')
  

  # Test double layer potential
  if True:
    print('TESTING STRESSLET LAYER POTENTIAL')
    # Init FMM
    #timer.timer('FMM_stresslet_init')
    #pbc = stkfmm.PAXIS.NONE
    #kernel = stkfmm.KERNEL.PVel
    #kernels_index = stkfmm.KERNEL(kernel)
    #fmm_PVel = stkfmm.STKFMM(mult_order, max_pts, pbc, kernels_index)
    #kdimSL, kdimDL, kdimTrg = stkfmm.getKernelDimension(fmm_PVel, kernel)
    #stresslet_kernel_source_target_stkfmm_partial = partial(kernels.stresslet_kernel_source_target_stkfmm, fmm_PVel=fmm_PVel)
    #timer.timer('FMM_stresslet_init')
    
    # Create source density 
    density_DL = np.random.randn(nsrc_DL, 3) 
    normals_DL = np.random.randn(nsrc_DL, 3) 
    
    # Call FMM first time
    timer.timer('FMM_stresslet_call_tree')
    v_stkfmm_set_tree = stresslet_kernel_source_target_stkfmm_partial(r_source_DL, r_target, normals_DL, density_DL)
    timer.timer('FMM_stresslet_call_tree')
    
    # Call FMM second time
    timer.timer('FMM_stresslet_call')
    v_stkfmm = stresslet_kernel_source_target_stkfmm_partial(r_source_DL, r_target, normals_DL, density_DL)
    timer.timer('FMM_stresslet_call')
    
    # Compute velocities with numba if system is not too large
    if np.sqrt((nsrc_DL) * ntrg) <= N_max:
      v_numba = kernels.stresslet_kernel_source_target_numba(r_source_DL, r_target, normals_DL, density_DL)
      timer.timer('numba_stresslet')
      v_numba = kernels.stresslet_kernel_source_target_numba(r_source_DL, r_target, normals_DL, density_DL)
      timer.timer('numba_stresslet')
      diff_0 = v_stkfmm_set_tree - v_numba
      diff_1 = v_stkfmm - v_numba
      
      # Compute errors
      print('\n\n')
      print('v_numba[0:6]          = ', v_numba[0:6])
      print('v_stkfmm[0:6]         = ', v_stkfmm[0:6])
      print('norm velocity         = ', np.linalg.norm(v_numba))
      print('relative L2 error (1) = ', np.linalg.norm(diff_0) / np.linalg.norm(v_numba))
      print('Linf error        (1) = ', np.linalg.norm(diff_0.flatten(), ord=np.inf))
      print('relative L2 error (2) = ', np.linalg.norm(diff_1) / np.linalg.norm(v_numba))
      print('Linf error        (2) = ', np.linalg.norm(diff_1.flatten(), ord=np.inf))
      print('\n\n\n')  


  # Test single and double layer potential
  if True:
    print('TESTING SINGLE AND DOUBLE LAYER POTENTIAL')
    # Init FMM
    timer.timer('FMM_SL_DL_init')
    pbc = stkfmm.PAXIS.NONE
    kernel = stkfmm.KERNEL.PVel
    kernels_index = stkfmm.KERNEL(kernel)
    fmm_PVel = stkfmm.STKFMM(mult_order, max_pts, pbc, kernels_index)
    kdimSL, kdimDL, kdimTrg = stkfmm.getKernelDimension(fmm_PVel, kernel)
    single_double_layer_kernel_source_target_stkfmm_partial = partial(kernels.single_double_layer_kernel_source_target_stkfmm, fmm_PVel=fmm_PVel)
    timer.timer('FMM_SL_DL_init')
    
    # Create source density 
    density_SL = np.random.randn(nsrc_SL, 3) 
    density_DL = np.random.randn(nsrc_DL, 3) 
    normals_DL = np.random.randn(nsrc_DL, 3) 
    
    # Call FMM first time
    timer.timer('FMM_SL_DL_call_tree')
    v_stkfmm_set_tree = single_double_layer_kernel_source_target_stkfmm_partial(r_source_SL, r_source_DL, r_target, normals_DL, density_SL, density_DL)
    timer.timer('FMM_SL_DL_call_tree')
    
    # Call FMM second time
    timer.timer('FMM_SL_DL_call')
    v_stkfmm = single_double_layer_kernel_source_target_stkfmm_partial(r_source_SL, r_source_DL, r_target, normals_DL, density_SL, density_DL)
    timer.timer('FMM_SL_DL_call')
    
    # Compute velocities with numba if system is not too large
    if np.sqrt((nsrc_SL + nsrc_DL) * ntrg) <= N_max:
      v_numba = kernels.oseen_kernel_source_target_numba(r_source_SL, r_target, density_SL)
      v_numba += kernels.stresslet_kernel_source_target_numba(r_source_DL, r_target, normals_DL, density_DL)
      timer.timer('numba_SL_DL')
      v_numba = kernels.oseen_kernel_source_target_numba(r_source_SL, r_target, density_SL)
      v_numba += kernels.stresslet_kernel_source_target_numba(r_source_DL, r_target, normals_DL, density_DL)
      timer.timer('numba_SL_DL')
      diff_0 = v_stkfmm_set_tree - v_numba
      diff_1 = v_stkfmm - v_numba
      
      # Compute errors
      print('\n\n')
      print('v_numba[0:6]          = ', v_numba[0:6])
      print('v_stkfmm[0:6]         = ', v_stkfmm[0:6])
      print('norm velocity         = ', np.linalg.norm(v_numba))
      print('relative L2 error (1) = ', np.linalg.norm(diff_0) / np.linalg.norm(v_numba))
      print('Linf error        (1) = ', np.linalg.norm(diff_0.flatten(), ord=np.inf))
      print('relative L2 error (2) = ', np.linalg.norm(diff_1) / np.linalg.norm(v_numba))
      print('Linf error        (2) = ', np.linalg.norm(diff_1.flatten(), ord=np.inf))
  


  timer.timer(' ', print_all = True)
