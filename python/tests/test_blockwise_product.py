from __future__ import division, print_function
import numpy as np


if __name__ == '__main__':
  I = np.eye(3)
  a = np.arange(4)

  print('I = \n', I)
  print('a = \n', a)

  aI = a[:, None, None] * I
  # aI = I * a[:, None, None]

  print('aI = \n', aI)
  print('aI.shape = ', aI.shape)
  aI = np.reshape(aI, (aI.size // 3, 3)).T
  print('aI = \n', aI)
  print('aI.shape = ', aI.shape)
  print('====================================================')
  print('\n\n\n')

  
  R = np.zeros((3, 12))
  R[:,0:3] = I
  R[:,3:6] = I*2
  R[:,6:9] = I*3
  R[:,9:] = I*4
  print('R = \n', R)
  print('a = \n', a)

  aR = np.array([a[i] * R[:,3*i:3*(i+1)] for i in range(len(a))])
  aR = np.reshape(aR, (aR.size // 3, 3)).T
  print('aR = \n', aR)
  print('aR.shape = ', aR.shape)


