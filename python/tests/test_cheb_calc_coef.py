'''
Test Chebyshev modes
'''
from __future__ import division, print_function
import numpy as np
import scipy.special as scspecial
import sys
sys.path.append('../')

from utils import cheb
from utils import timer


def func(s):
  return np.exp(-s**2) + np.cos(np.pi * s * 3)


if __name__ == '__main__':
  N = 32
  m = 12

  D, s = cheb.cheb(N-1)

  x = func(s)

  for i in range(m):
    timer.timer('cheb_calc_coef_loops')
    modes = cheb.cheb_calc_coef_loops(x)
    timer.timer('cheb_calc_coef_loops')

    timer.timer('cheb_calc_coef')
    modes_new = cheb.cheb_calc_coef(x)
    timer.timer('cheb_calc_coef')

  diff = modes_new - modes
  error_L2 = np.linalg.norm(diff) / np.linalg.norm(modes)
  error_Linf = np.linalg.norm(diff, ord=np.inf) / np.linalg.norm(modes)

  print('error_L2   = ', error_L2)
  print('error_Linf = ', error_Linf, '\n')

  for i in range(m):
    s0 = np.random.rand(10) * 2.0 - 1.0
    x_th = func(s0)
    timer.timer('cheb_eval')
    x_modes = cheb.cheb_eval(s0, modes)
    timer.timer('cheb_eval')
    timer.timer('cheb_eval_loops')
    x_modes_loops = cheb.cheb_eval_loops(s0, modes)
    timer.timer('cheb_eval_loops')
    diff = x_modes - x_th 
    diff_loops = x_modes_loops - x_th 
    print('error       = ', np.linalg.norm(diff))
    print('error_loops = ', np.linalg.norm(diff_loops))


  timer.timer('', print_all=True)




