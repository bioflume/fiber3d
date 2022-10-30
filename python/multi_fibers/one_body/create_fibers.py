from __future__ import division, print_function
import numpy as np
import sys
sys.path.append('../')

from fiber import fiber


if __name__ == '__main__':

  # Create fiber
  form = 'straight'
  num = 1
  offset_x = 0.0
  offset_y = 0.0
  offset_z = 1.0
  E = 1.0
  length = 5.0
  num_points = 32
  dt = 1.0
  fib = fiber.fiber(num_points = num_points, dt = dt, E=E, length = length)


  print(num)
  for i in range(num):
    if form == 'straight':
      fib.x[:,0] = offset_x
      fib.x[:,1] = offset_y
      fib.x[:,2] = -(length / 2.0 * fib.s) + 0.5 * length + offset_z
    elif form == 'straight_wiggled':
      fib.x[:,0] = np.random.randn(num_points) * 1e-05
      fib.x[:,1] = np.random.randn(num_points) * 1e-05
      fib.x[:,2] = (length / 2.0 * fib.s) + 0.5 * length 
    elif form == 'semicircle':
      fib.x[:,0] = (length / np.pi) * np.cos(fib.s * np.pi / 2.0)
      fib.x[:,1] = (length / np.pi) * np.sin(fib.s * np.pi / 2.0)
      fib.x[:,2] = i * offset_z
    else:
      print('# Error')
    
    # Print coordinates
    print(num_points, E, length)
    np.savetxt(sys.stdout, fib.x)




