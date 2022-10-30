from __future__ import division, print_function
import numpy as np
import sys
sys.path.append('../shape_gallery/')

from shape_gallery import shape_gallery


if __name__ == '__main__':
  N = 800
  nodes, normal, h, gradh = shape_gallery('ellipsoid', N, radius = 2.0, a=1.0, b=1.0, c=1.0)
  nodes_and_normal = np.hstack([nodes, normal])

  print(N)
  np.savetxt('kk.xyz', nodes, header=str(N), comments='')
  np.savetxt('kk.dat', normal, header=str(N), comments='')
  np.savetxt('kkk.xyz', nodes_and_normal, header=str(N), comments='')


  

