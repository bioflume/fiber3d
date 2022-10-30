'''
 Example to test smooth forces.
'''
# Standard imports
from __future__ import print_function
import numpy as np
import scipy.sparse.linalg as spla
import scipy.linalg 
import scipy.optimize as scop
from functools import partial
import sys

sys.path.append('../')

# import nonlin

# Local imports
from fiber import fiber
from utils import cheb
from force_generator import force_generator as fg
from utils import timer
from utils import nonlinear

class gmres_counter(object):
  '''
  Callback generator to count iterations. 
  '''
  def __init__(self, print_residual = False):
    self.print_residual = print_residual
    self.niter = 0
  def __call__(self, rk=None):
    self.niter += 1
    if self.print_residual is True:
      if self.niter == 1:
        print( 'gmres =  0 1')
      print( 'gmres = ', self.niter, rk)

    
if __name__ == '__main__':

  # Set some parameters 
  max_steps = 200
  n_save = 10
  name_output = 'data/run'
  num_points = 32
  dt = 1e-02
  tolerance = 1e-6
  max_iterations = 100
  print_residual = True
  # Select method to solve linear system 'dense_algebra', 'iterative_block'
  method = 'dense_algebra'

  # Create fiber
  E = 1.0
  length = 5.0
  fib = fiber.fiber(num_points = num_points, dt = dt, E=E, length = length)

  # Set fiber like semi-circle
  fib.x[:,0] = (length / np.pi) * np.cos(fib.s * np.pi / 2.0)
  fib.x[:,1] = (length / np.pi) * np.sin(fib.s * np.pi / 2.0)
  
  # Set fiber like a spiral 
  #fib.x[:,0] = 1 * np.cos(4 * fib.s)
  #fib.x[:,1] = 1 * np.sin(4 * fib.s)
  #fib.x[:,2] = np.sqrt(100.0 / 64.0 - 1.0) * fib.s / 0.25

  # Init some variables
  x = np.zeros(4 * num_points + 12)
  x[0*num_points:1*num_points] = fib.x[:,0]
  x[1*num_points:2*num_points] = fib.x[:,1]
  x[2*num_points:3*num_points] = fib.x[:,2]

  # Loop over time step
  timer.timer('zzz_loop')
  for step in range(max_steps):
    # Compute material derivative
    xs = np.dot(fib.D_1, fib.x)
    timer.timer('save_data')
    if (step % n_save) == 0:
      print( 'step = ', step)
      mode = 'a'
      if step == 0:
        mode = 'w'
      name = name_output + '.config'
      with open(name, mode) as f:
        f.write(str(num_points) + '\n')
        y = np.empty((num_points, 4))
        y[:,0:3] = fib.x
        y[:,3] = x[3*num_points:4*num_points]
        np.savetxt(f, y)
      # print( y)     
      stretching_error = np.sqrt(xs[:,0]**2 + xs[:,1]**2 + xs[:,2]**2) - 1.0
      stretching_max_error = max(stretching_error, key=abs)
      name = name_output + '.stretching_error.dat'
      with open(name, mode) as f:
        f.write(str(num_points) + '\n')
        np.savetxt(f, stretching_error)
      name = name_output + '.stretching_max_error.dat'
      with open(name, mode) as f:
        f.write(str(stretching_max_error) + '\n')
    timer.timer('save_data')


    print('===================================================== ')
    x[3*num_points:] = 0.0
    # x[0*num_points:1*num_points] += 0.1
    # for i in range(x.size):
    #   epsilon = 1e-06
    #   dx = np.zeros(x.size)
    #   dx[i] = epsilon
    #   J = fib.self_jacobian_force(x)
    #   df = (fib.self_residual_force(x + dx) - fib.self_residual_force(x)) / epsilon
    #   df_linear = np.dot(J, dx) / epsilon
    #   print(i, 'error = ', np.linalg.norm(df - df_linear, ord=np.inf))
    #   # print (df - df_linear) 
    (sol, res_norm, it, nonlinear_evaluations, gmres_iterations) = nonlinear.nonlinear_solver(fib.self_residual_force,
                                                                                              x,
                                                                                              fib.self_jacobian_force,
                                                                                              M = fib.preconditioner_jacobian,
                                                                                              verbose = True,
                                                                                              gmres_verbose = False,
                                                                                              max_outer_it = 20,
                                                                                              max_inner_it = 10,
                                                                                              tol = tolerance,
                                                                                              gmres_restart = 100)
    print('===================================================== ')

    # Update fiber configuration
    x = sol
    fib.x[:,0] = sol[0*num_points : 1*num_points]
    fib.x[:,1] = sol[1*num_points : 2*num_points]
    fib.x[:,2] = sol[2*num_points : 3*num_points]
  timer.timer('zzz_loop')

  # Save last configuration as an edge case
  timer.timer('save_data')
  if (max_steps % n_save) == 0:
    print( 'step = ', max_steps)
    mode = 'a'
    if max_steps == 0:
      mode = 'w'
    name = name_output + '.config'
    with open(name, mode) as f:
      f.write(str(num_points) + '\n')
      y = np.empty((num_points, 4))
      y[:,0:3] = fib.x
      y[:,3] = x[3*num_points:4*num_points]
      np.savetxt(f, y)
      # print( y)
    xs = np.dot(fib.D_1, fib.x)
    stretching_error = np.sqrt(xs[:,0]**2 + xs[:,1]**2 + xs[:,2]**2) - 1.0
    stretching_max_error = max(stretching_error, key=abs)
    name = name_output + '.stretching_error.dat'
    with open(name, mode) as f:
      f.write(str(num_points) + '\n')
      np.savetxt(f, stretching_error)
    name = name_output + '.stretching_max_error.dat'
    with open(name, mode) as f:
      f.write(str(stretching_max_error) + '\n')
  timer.timer('save_data')

  print( 'beta = %.4g' % fib.beta)

  timer.timer(None, print_all = True)

  print( '# Main End')
