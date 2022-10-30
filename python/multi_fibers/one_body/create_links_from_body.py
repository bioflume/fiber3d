from __future__ import division, print_function
import numpy as np
import sys



if __name__ == '__main__':
  name = sys.argv[1]

  # Define center and spring constants
  center = np.array([0.0, 0.0, 0.0])
  spring_constant = 1.0
  spring_constant_angle = 1.0

  # Load locations
  r_vectors = np.loadtxt(name, skiprows=1)
  r_vectors = r_vectors.reshape(r_vectors.size // 3, 3)
  # N = r_vectors.shape[0]
  N = 1

  # Loop over locations and print link
  print(N)
  count = 0
  for i, r in enumerate(r_vectors):
    # if i < N:
    if r[2] < 0 and count < N:
      r_norm = np.linalg.norm(r)
      axis = r / r_norm
      print(spring_constant, ' ', spring_constant_angle, ' ', end='')
      np.savetxt(sys.stdout, r, newline=' ') 
      np.savetxt(sys.stdout, axis, newline=' ')
      print('')
      count += 1

