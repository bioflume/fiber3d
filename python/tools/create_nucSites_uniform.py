from __future__ import division, print_function
import numpy as np
import sys

if __name__ == '__main__':
  # Read parameters
  np.random.seed(None)
  num_points = int(sys.argv[1])
  radius = float(sys.argv[2])
  name = sys.argv[3]

  theta = 2 * np.pi * np.random.rand(num_points)
  phi = np.arccos(1-2*np.random.rand(num_points))

  x = radius * np.sin(phi) * np.cos(theta)
  y = radius * np.sin(phi) * np.sin(theta)
  z = radius * np.cos(phi)


  # Save last configuration
  with open(name, 'w') as f:
    #f.write(str(N) + '\n# \n')
    for i in range(num_points):
      f.write(str(0) + '  ' + str(x[i]) + '  ' + str(y[i]) + '  ' + str(z[i]) + '\n')
