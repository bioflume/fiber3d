'''
Test Aliasing of non-linear functions.
'''
from __future__ import division, print_function
import numpy as np
import sys
sys.path.append('../')

from utils import cheb
from utils import barycentricMatrix as bary
from utils import timer


def func(s):
  return np.exp(-s**2) * np.sin(np.pi * s * 1.333) + np.cos(np.pi * s * 3)



if __name__ == '__main__':
  # Define parameters
  N = 24
  upsampling = 1.5
  N_up = int(N * upsampling)

  # Get cheb points 
  D, s = cheb.cheb(N-1)
  s = np.flipud(s)
  D, s_up = cheb.cheb(N_up - 1)
  s_up = np.flipud(s_up)
  
  # Get interpolation matrices
  P_up = bary.barycentricMatrix(s, s_up)
  P_down = bary.barycentricMatrix(s_up, s)

  # Get filters
  P_rTF = cheb.cheb_real_to_reciprocal(N_up)
  P_FTr = cheb.cheb_reciprocal_to_real(N)
  I_NNup = np.eye(N, N_up)
  P_filter = np.dot(P_FTr, np.dot(I_NNup, P_rTF))

  # Compute functions and product
  f = func(s)
  g = func(s)
  fg = f * g

  # Compute product with up-down sampling
  f_up = np.dot(P_up, f)
  g_up = np.dot(P_up, g)
  fg_up = f_up * g_up
  fg_up_down = np.dot(P_down, fg_up)

  # Compute product with filtering
  f_up = np.dot(P_up, f)
  g_up = np.dot(P_up, g)
  fg_up = f_up * g_up
  fg_filter = np.dot(P_filter, fg_up)

  # Compute modes
  f_modes = cheb.cheb_calc_coef(f)
  f_up_modes = cheb.cheb_calc_coef(f_up)
  fg_modes = cheb.cheb_calc_coef(fg)
  fg_up_modes = cheb.cheb_calc_coef(fg_up)
  fg_up_down_modes = cheb.cheb_calc_coef(fg_up_down)
  fg_filter_modes = cheb.cheb_calc_coef(fg_filter)

  # Save results
  result_N = np.zeros((N,9))
  result_N[:,0] = s
  result_N[:,1] = f
  result_N[:,2] = fg
  result_N[:,3] = fg_up_down
  result_N[:,4] = fg_filter
  result_N[:,5] = abs(f_modes)
  result_N[:,6] = abs(fg_modes)
  result_N[:,7] = abs(fg_up_down_modes)
  result_N[:,8] = abs(fg_filter_modes)
  np.savetxt('kk.' + str(N) + '.dat', result_N)

  result_N_up = np.zeros((N_up,4))
  result_N_up[:,0] = s_up
  result_N_up[:,1] = f_up
  result_N_up[:,2] = fg_up
  result_N_up[:,3] = abs(fg_up_modes)
  np.savetxt('kk_up.' + str(N) + '.dat', result_N_up)
  
