'''
 Example to test smooth forces.
'''
# Standard imports
from __future__ import print_function
import numpy as np
import scipy.sparse.linalg as spla
import scipy.linalg 
from functools import partial
import sys

sys.path.append('../')

# Local imports
from fiber import fiber
from utils import cheb
from force_generator import force_generator as fg
from utils import timer

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
  max_steps = 10000
  n_save = 100
  name_output = 'data/run_new'
  num_points = 64
  dt = 1e-3
  tolerance = 1e-16
  print_residual = True
  # Select method to solve linear system 'dense_algebra', 'iterative_block'
  method = 'dense_algebra'

  # Create fiber
  fib = fiber.fiber(num_points = num_points, dt = dt, E=1, length = 10.0)
  
  # Set initial configuration
  fib.x[:,0] = fib.s * (fib.length / 2.0) * np.cos(0.1) + 4
  fib.x[:,1] = fib.s * (fib.length / 2.0) * np.sin(0.1) + 0.1
  # fib.x[:,0] = 5.0
  # fib.x[:,1] = fib.s * (fib.length / 2.0) 

  # Create force generator, square lattice on the plane z=0
  spring_constant = 1.0
  active_force = 1.0
  unbind_period = 1.0
  radius = 0.50
  num_lattice = 10
  L_lattice = 10
  grid_x = np.linspace(0, L_lattice, num_lattice)
  grid_y = np.linspace(0, L_lattice, num_lattice)
  yy, xx = np.meshgrid(grid_y, grid_x, indexing = 'ij')
  r_fg = np.zeros((xx.size, 3))
  r_fg[:,0] = np.reshape(xx, xx.size) - L_lattice * 0.5 + 5.0
  r_fg[:,1] = np.reshape(yy, yy.size) - L_lattice * 0.5 + 5.0
  fg_0 = fg.force_generator(r_fg, radius = radius, spring_constant = spring_constant, active_force = active_force, unbind_period = unbind_period, alpha = 0.2)
  name = name_output + '.force_generator'
  with open(name, 'w') as f:
    f.write(str(fg_0.r.size / 3) + '\n')
    np.savetxt(f, fg_0.r)

  # Init some variables
  sol = np.zeros(num_points * 4)
  x0 = np.zeros(4 * num_points)
  x0[0:num_points] = fib.x[:,0]
  x0[num_points:2*num_points] = fib.x[:,1]
  x0[2*num_points:3*num_points] = fib.x[:,2]
  sol = x0

  # Loop over time step
  timer.timer('zzz_loop')
  for step in range(max_steps):
    # Compute material derivative
    xs = np.dot(fib.D_1, fib.x)

    # Save info
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
        y[:,3] = sol[3*num_points:]
        np.savetxt(f, y)
      print( y)     
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

    # Create external density force
    # 1. Unbind microtubules
    timer.timer('fg_release')
    fg_0.release_particle(fib.x, np.array([num_points]), np.array([0]), dt)
    timer.timer('fg_release')
    # 2. Capture microtubules
    timer.timer('fg_trap')
    fg_0.trap_particle(fib.x, 1, np.array([num_points]), np.array([0]))
    timer.timer('fg_trap')
    # 3. Compute active and and passive force
    timer.timer('fg_force')
    force = fg_0.compute_force_pycuda(fib.x, xs, 1, np.array([num_points]), np.array([0]))
    # force = np.zeros((num_points, 3))
    timer.timer('fg_force')
    
    # Set Boundary conditions
    # fib.set_BC(BC_end_0 = 'position', BC_end_vec_0 = np.array([5.0 + 3.0 * np.sin(step * dt), 0.5 * fib.length, 0.0]))

    # Get linear operator, RHS and apply BC
    timer.timer('A')
    A = fib.form_linear_operator()
    timer.timer('A')
    timer.timer('RHS')
    RHS = fib.compute_RHS(force_external = force)
    timer.timer('RHS')
    timer.timer('BC')
    A, RHS = fib.apply_BC(A, RHS)
    timer.timer('BC')
 
    # Solve linear system 
    timer.timer('solver')
    if method == 'dense_algebra':
      sol = np.linalg.solve(A, RHS) 
    elif method == 'iterative_block':
      counter = gmres_counter(print_residual = print_residual) 
      (LU, P) = scipy.linalg.lu_factor(A)
      def P_inv(LU, P, x):
        return scipy.linalg.lu_solve((LU, P), x)
      P_inv_partial = partial(P_inv, LU, P)
      P_inv_partial_LO = spla.LinearOperator((4*num_points, 4*num_points), matvec = P_inv_partial, dtype='float64')
      (sol, info_precond) = spla.gmres(A, RHS, x0=x0, tol=tolerance, M=P_inv_partial_LO, maxiter=1000, restart=150, callback=counter) 
      x0 = sol   
    timer.timer('solver')      

    # Update fiber configuration
    fib.x[:,0] = sol[0:num_points]
    fib.x[:,1] = sol[num_points:2*num_points]
    fib.x[:,2] = sol[2*num_points:3*num_points]         
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
      y[:,3] = sol[3*num_points:]
      np.savetxt(f, y)
      print( y)
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
