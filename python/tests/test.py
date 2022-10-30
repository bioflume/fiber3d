from __future__ import division, print_function
import numpy as np


if __name__ == '__main__':
  I = np.arange(9)
  I = np.reshape(I, (3,3))


  print('I                 = \n', I, '\n')
  print('np.sum(I)         = ', np.sum(I))
  print('np.sum(I, axis=0) = ', np.sum(I, axis=0))
  print('np.sum(I, axis=1) = ', np.sum(I, axis=1))
  

