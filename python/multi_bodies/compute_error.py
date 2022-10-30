from __future__ import division, print_function
import numpy as np
import sys

if __name__ == '__main__':
  
  # space unbounded or shell
  # space = 'shell' 
  # space = 'unbounded'
  space = 'squirmer'
  name = sys.argv[1]
  v = np.loadtxt(name)
   
  
  if space == 'unbounded': 
    a = 1. 
    v_th = np.array([1.0 / (6.0 * np.pi * a), 0.0, 0.0, 0.0, 0.0, 1.0 / (8.0 * np.pi * a**3)]) 
  elif space == 'shell':
    T = 0
    a = 0.2
    b = 2.0
    l = a/b
    alpha = 1 - 2.25*l + 2.5*l**3 - 2.25*l**5 + l**6
    K = (1 - l**5) / alpha
    F = 6 * np.pi * a * K
    v_th = np.array([0., 0., -F / (6*np.pi*a*K), 0., 0., 0.])
    print('F(V=1)   = ', 6*np.pi*a*K, '\n\n')
  elif space == 'squirmer':
    v_th = np.array([0., 0., 2.0/3.0, 0., 0., 0.])

  
  error = v - v_th
  error_norm = np.linalg.norm(error) / np.linalg.norm(v_th)


  print('v_th     = ', v_th)
  print('v      = ', v, '\n\n')
  print('error_norm = ', error_norm)

