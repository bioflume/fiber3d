from __future__ import division, print_function
import numpy as np
import sys



if __name__ == '__main__':
  name = sys.argv[1]

  # Define center and spring constants
  center = np.array([0.0, 0.0, 0.0])
  spring_constant = 10.0
  spring_constant_angle = 1.0

  # Load locations
  r_vectors = np.loadtxt(name, skiprows=0)
  N = r_vectors.shape[0]

  # Loop over locations and print link
  print(N)
  count = 0
  for i, r in enumerate(r_vectors):
    # if i < N:
    if count < N:
      if r[3]>0:
        r_norm = np.linalg.norm(r[1:]-np.array([0, 0, 4.44]))
        axis = r[1:]-np.array([0, 0, 4.44])
        axis = axis / r_norm
      else:
        r_norm = np.linalg.norm(r[1:]-np.array([0, 0, -4.44]))
        axis = r[1:] - np.array([0, 0, -4.44])
        axis = axis / r_norm
      
      #r_norm = np.linalg.norm(r)
      #axis = r / r_norm
      print(spring_constant, ' ', spring_constant_angle, ' ', end='')
      np.savetxt(sys.stdout, r[1:], newline=' ') 
      np.savetxt(sys.stdout, axis, newline= ' ')
      #np.savetxt(sys.stdout, axis[1:], newline=' ')
      print('')
      count += 1

