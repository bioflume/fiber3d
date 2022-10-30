from __future__ import division, print_function
import numpy as np
import sys

if __name__ == '__main__':
  # Read parameters
  num_points = int(sys.argv[1])
  radius = float(sys.argv[2])


  ga = (3 - np.sqrt(5)) * np.pi # golden angle

  # Create a list of golden angle increments along tha range of number of points
  theta = ga * np.arange(num_points)

  # Z is a split into a range of -1 to 1 in order to create a unit circle
  z = np.linspace(1/num_points-1, 1-1/num_points, num_points)

  # a list of the radii at each height step of the unit circle
  radius = np.sqrt(1 - z * z)

  # Determine where xy fall on the sphere, given the azimuthal and polar angles
  y = radius * np.sin(theta)
  x = radius * np.cos(theta)


  # Save last configuration
  name = sys.argv[3]
  with open(name, 'w') as f:
    #f.write(str(N) + '\n# \n')
    for i in range(num_points):
      f.write(str(0) + '  ' + str(x[i]) + '  ' + str(y[i]) + '  ' + str(z[i]) + '\n')
