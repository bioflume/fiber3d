'''
Test quadrature methods: trapezoidal, Clenshaw-Curtis, Chebyshev-Gauss and
Gauss. 
'''
from __future__ import division, print_function
import numpy as np
import scipy.special as scspecial
import sys
sys.path.append('../')

from utils import cheb





if __name__ == '__main__':

  M = 100

  def func(x):
    return np.exp(-x**2) 
  I_th = np.sqrt(np.pi) * scspecial.erf(1.0) 

  print('# Columns: number of points, value theory, error trapezoidal, ...')
  for N in range(2, M):
    # Prepare Chebyshev points
    D, s_cheb = cheb.cheb(N-1)
    w_cheb = cheb.clencurt(N-1)
    s_gauss, w_gauss = cheb.gauss_weights(N)
    s_roots = cheb.cheb_root_points(N)
    w_roots = (np.pi / N) * np.sqrt(1.0 - s_roots**2)
    f_cheb = func(s_cheb)
    f_roots = func(s_roots)
    f_gauss = func(s_gauss)

    # Compute integrals
    I_cheb = sum(f_cheb * w_cheb)
    I_trap = np.trapz(f_cheb, x=-s_cheb)
    I_gauss = sum(f_gauss * w_gauss)
    I_roots = sum(f_roots * w_roots)
    print(N, I_th, abs(I_trap - I_th), abs(I_cheb - I_th), abs(I_roots - I_th), abs(I_gauss - I_th))
    # print(N, I_gauss, abs(I_trap - I_gauss), abs(I_cheb - I_gauss), abs(I_roots - I_gauss))

  
