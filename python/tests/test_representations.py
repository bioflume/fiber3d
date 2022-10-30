'''
Test to transform information between the real and the
reciprocal spaces.
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
  print('# Start')

  # Define parameters
  N = 24

  # Get cheb points 
  D, s = cheb.cheb(N-1)
  flip = 1
  if flip:
    s = np.flipud(s)

  # Compute functions in real space
  f = func(s)

  # Compute modes
  f_m = cheb.cheb_calc_coef(f)
  P_real_to_recip = cheb.cheb_real_to_reciprocal(f.size)
  f_m_P = np.dot(P_real_to_recip, f)
  diff = f_m - f_m_P
  print('error = ', np.linalg.norm(diff))
  print('relative error = ', np.linalg.norm(diff) / np.linalg.norm(f_m))
  print('\n')

  
  if False:
    print('\n')
    print(f_m)
    print('\n')
    print(f_m_P)

  
  # Compute real values using modes
  f_r = cheb.cheb_eval(s, f_m, order = flip)
  P = cheb.cheb_reciprocal_to_real(f.size)
  f_r_P = np.dot(P, f_m)

  diff = f_r - f
  print('error = ', np.linalg.norm(diff))
  print('relative error = ', np.linalg.norm(diff) / np.linalg.norm(f))
  print('\n')


  if False:
    print('\n')
    print(f)
    print('\n')
    print(f_r)
    print('\n')
    print(f_r_P)
  
  


