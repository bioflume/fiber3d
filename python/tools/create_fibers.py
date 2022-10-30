from __future__ import division, print_function
import numpy as np
import sys
sys.path.append('../')
sys.path.append('../../')

from fiber import fiber


if __name__ == '__main__':

  # Create fiber
  form = 'straight'
  num = 1
  offset_x = 0.0
  offset_y = 0.0
  offset_z = 0.15
  alpha = 0.1
  E = 1.0
  length = 1.85
  num_points = 32
  dt = 1.0
  fib = fiber.fiber(num_points = num_points, dt = dt, E=E, length = length)

  # The first point of the fiber is where the fiber is attached to a body,
  # so make sure that link, fiber, body are consistent.
  print(num)
  for i in range(num):
    if form == 'straight':
      fib.x[:,0] = offset_x
      fib.x[:,1] = offset_y
      #fib.x[:,2] = -(length / 2.0 * fib.s) + 0.5 * length + offset_z
      fib.x[:,2] = np.flipud(fib.s + offset_z)
    elif form == 'angle':
      fib.x[:,0] = -(length / 2.0 * fib.s) * np.sin(alpha)
      fib.x[:,1] = offset_y
      fib.x[:,2] = -(length / 2.0 * fib.s) * np.cos(alpha) + 0.5 * length + offset_z
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




