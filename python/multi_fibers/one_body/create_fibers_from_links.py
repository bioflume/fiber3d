from __future__ import division, print_function
import numpy as np
import sys
sys.path.append('../')
sys.path.append('../../')

from fiber import fiber
from utils import cheb


if __name__ == '__main__':
  name = sys.argv[1]
  
  # Parameters
  num_points = 24
  E = 1.0
  length0 = 1.0
  center = np.array([0., 0., 3.0])

  # Get Chebyshev differential matrix
  D_1, s = cheb.cheb(num_points - 1)
  s = np.flipud(s)
  s += 1.0
  
  # Load links
  links = np.loadtxt(name, skiprows=1)
  # N = links.shape[0]
  N = links.size // 8
  if N == 1:
    links = np.reshape(links, (1, 8))
  
  # Loop over links
  print(N)
  for link in links:
    length = length0 # + np.random.randn(1)

    # Get location and axis
    location = link[2:5] 
    axis = link[5:]

    # Ge fiber configuration
    axis_s = np.empty((s.size, 3))
    axis_s[:,0] = axis[0] * s
    axis_s[:,1] = axis[1] * s
    axis_s[:,2] = axis[2] * s
    axis_s = axis_s * (length / 2.0) + location + center

    # Print configuration
    print(num_points, E, length)
    np.savetxt(sys.stdout, axis_s)


