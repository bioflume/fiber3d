from __future__ import division, print_function
import numpy as np
import sys
sys.path.append('../')

from utils import cheb

if __name__ == '__main__':

  # Set parameters
  N = 32
  x = 0.0
  F = 1.0
  sigma = 0.25
  L = 5.0
  
  
  D, s_0 = cheb.cheb(N-1)
  s = s_0 * (L / 2.0)
  w_0 = cheb.clencurt(N-1)
  w = w_0 * (L / 2.0)

  def kernel(x, F, s, sigma):
    return F * np.exp(-(s-x)**2 / (2*sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
    # return F * (np.exp(-(s-x-s[0])**2 / (2*sigma**2)) + np.exp(-(s-x+s[-1])**2 / (2*sigma**2)) + np.exp(-(s-x)**2 / (2*sigma**2))) / np.sqrt(2 * np.pi * sigma**2)


  f = kernel(x, F, s, sigma)
  FI = sum(f * w)

  print('x  = ', x)
  print('F  = ', F)
  print('FI = ', FI)
  print('f  = \n', f)

