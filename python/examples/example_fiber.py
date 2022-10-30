# Standard imports
import numpy as np
import scipy.sparse.linalg as spla
import scipy.linalg 
from functools import partial
import sys

sys.path.append('../')

# Local imports
from fiber import fiber
from utils import cheb


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
        print 'gmres =  0 1'
      print 'gmres = ', self.niter, rk

    
if __name__ == '__main__':

  # Set some parameters
  max_steps = 10
  n_save = 1
  name_output = 'run'
  num_points = 32
  dt = 1e-3
  tolerance = 1e-16
  print_residual = True
  # Select method to solve linear system 'dense_algebra', 'iterative_block'
  method = 'dense_algebra'

  # Create fiber
  fib = fiber.fiber(num_points = num_points, dt = dt, E=1)
  
  # Set initial configuration
  fib.x[:,0] = 0
  fib.x[:,1] = -fib.s * (fib.length / 2.0) + 0.5 

  # Init some variables
  sol = np.zeros(num_points * 4)
  x0 = np.zeros(4 * num_points)
  x0[0:num_points] = fib.x[:,0]
  x0[num_points:2*num_points] = fib.x[:,1]
  x0[2*num_points:3*num_points] = fib.x[:,2]
  sol = x0

  # Loop over time step
  for step in range(max_steps):
    # Save info
    if (step % n_save) == 0:
      print 'step = ', step
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
      print y
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

    # Create flow 
    flow = np.zeros((num_points, 3))

    # Create external density force
    force = np.zeros((num_points, 3))

    # Set Boundary conditions
    # Example 1, one end fixed at (0,0,0) and with orientation (0,1,0)
    # fib.set_BC(BC_end_0 = 'position', BC_end_vec_0 = np.zeros(3), BC_end_1 = 'angle', BC_end_vec_1 = np.array([0.0, -1.0, 0.0]))
    # Example 2, one end subject to a force F=(sin(10 * step*dt), 0, 0) and the other F=(0, 0, sin(10 * step*dt))
    # F_start = np.array([np.sin(10 * step*dt), 0, 0])
    # F_end = np.array([0, 0, np.sin(10 * step*dt)])
    # fib.set_BC(BC_start_vec_0 = F_start, BC_end_vec_0 = F_end)
    # Example 3, one end fixed at (0,0,0) and a force (0.01, -1, 0) on the other end
    # fib.set_BC(BC_start_0 = 'position', BC_start_vec_0 = np.zeros(3), BC_end_vec_0 = np.array([0.1, -1.0, 0.0]))

    # Get linear operator, RHS and apply BC
    A = fib.form_linear_operator()
    RHS = fib.compute_RHS(flow = flow, force_external = force)
    A, RHS = fib.apply_BC(A, RHS)
 
    # Solve linear system 
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
      
    # Update fiber configuration
    fib.x[:,0] = sol[0:num_points]
    fib.x[:,1] = sol[num_points:2*num_points]
    fib.x[:,2] = sol[2*num_points:3*num_points]         

  # Save last configuration as an edge case
  if (max_steps % n_save) == 0:
    print 'step = ', max_steps
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
      print y
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



  print '# Main End'
