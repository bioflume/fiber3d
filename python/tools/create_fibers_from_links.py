from __future__ import division, print_function
import numpy as np
import sys
sys.path.append('../')
sys.path.append('../../')

from fiber import fiber
from utils import cheb
from quaternion.quaternion import Quaternion


if __name__ == '__main__':
  name_clones = sys.argv[1]
  name_links = sys.argv[2]
  
  # Parameters
  num_points = 32
  E = 10.0
  length0 = float(sys.argv[3])


  # Read clones (assume one clone)
  x_clones = np.loadtxt(name_clones, skiprows=1)
  x_clones = x_clones.reshape(x_clones.size // 7, 7)
  center = x_clones[0, 0:3]
  orientation = Quaternion(x_clones[0, 3:])
  rotation_matrix = orientation.rotation_matrix()
  

  # Get Chebyshev differential matrix
  #D_1, s = cheb.cheb(num_points - 1)
  s = np.linspace(0,2,num_points)
  #s = np.flipud(s)
  #s += 1.0
  
  # Load links
  links = np.loadtxt(name_links, skiprows=1)
  # N = links.shape[0]
  N = links.size // 8
  if N == 1:
    links = np.reshape(links, (1, 8))
  
  # Loop over links
  print(N)
  for link in links:
    length = length0 # + np.random.randn(1)

    # Get location and axis
    location = np.dot(rotation_matrix, link[2:5])
    axis = np.dot(rotation_matrix, link[5:])

    # Ge fiber configuration
    axis_s = np.empty((s.size, 3))
    axis_s[:,0] = axis[0] * s
    axis_s[:,1] = axis[1] * s
    axis_s[:,2] = axis[2] * s
    axis_s = axis_s * (length / 2.0) + location + center

    # Print configuration
    print(num_points, E, length)
    np.savetxt(sys.stdout, axis_s)


