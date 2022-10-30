'''
 Example to test smooth forces.
'''
# Standard imports
from __future__ import print_function
import numpy as np
import scipy.sparse.linalg as spla
import scipy.linalg 
import scipy
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
  max_steps = 100000
  n_save = 100
  name_output = 'data/run305'
  num_points = 64
  dt = 5e-4
  tolerance = 1e-16
  print_residual = False
  # Select method to solve linear system 'dense_algebra', 'iterative_block'
  method = 'iterative_block'

  # Create fiber
  fib_0 = fiber.fiber(num_points = num_points, dt = dt, E=1, length = 10.0)
  fib_1 = fiber.fiber(num_points = num_points, dt = dt, E=1, length = 10.0)
  
  # Set initial configuration
  fib_0.x[:,0] = fib_0.s * (fib_0.length / 2.0) * np.cos(0.1) + 4
  fib_0.x[:,1] = fib_0.s * (fib_0.length / 2.0) * np.sin(0.1) + 0.1
  fib_1.x[:,0] = fib_1.s * (fib_1.length / 2.0) * np.cos(0.5) + 3.41289198306173
  fib_1.x[:,1] = fib_1.s * (fib_1.length / 2.0) * np.sin(0.5) + 1.99796060978687

  angle_0_1 = (np.dot(fib_0.D_1, fib_0.x) - np.dot(fib_1.D_1, fib_1.x))[0,:]
  angle_0_1.flatten()

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
  sol = np.zeros(num_points * 4 * 2)
  x0 = np.zeros(4 * num_points * 2)
  x0[0:num_points] = fib_0.x[:,0]
  x0[num_points:2*num_points] = fib_0.x[:,1]
  x0[2*num_points:3*num_points] = fib_0.x[:,2]
  x0[4*num_points:5*num_points] = fib_1.x[:,0]
  x0[5*num_points:6*num_points] = fib_1.x[:,1]
  x0[6*num_points:7*num_points] = fib_1.x[:,2]
  sol = x0

  # Loop over time step
  timer.timer('zzz_loop')
  for step in range(max_steps):
    # Compute material derivative
    xs = []
    xs.append( np.dot(fib_0.D_1, fib_0.x) )
    xs.append( np.dot(fib_1.D_1, fib_1.x) )
    xs = np.array(xs)
    xs = np.reshape(np.array(xs), (xs.size / 3, 3))

    # Save info
    timer.timer('save_data')
    if (step % n_save) == 0:
      print( 'step = ', step)
      mode = 'a'
      if step == 0:
        mode = 'w'
      name = name_output + '.config'
      with open(name, mode) as f:
        f.write(str(num_points * 2) + '\n')
        y = np.empty((num_points * 2, 4))
        y[0:num_points, 0:3] = fib_0.x
        y[0:num_points,3] = sol[3*num_points:4*num_points]
        y[num_points:2*num_points, 0:3] = fib_1.x
        y[num_points:2*num_points,3] = sol[7*num_points:8*num_points]
        np.savetxt(f, y)
      print( y)     
      stretching_error = np.sqrt(xs[:,0]**2 + xs[:,1]**2 + xs[:,2]**2) - 1.0
      stretching_max_error = max(stretching_error, key=abs)
      name = name_output + '.stretching_error.dat'
      with open(name, mode) as f:
        f.write(str(num_points * 2) + '\n')
        np.savetxt(f, stretching_error)
      name = name_output + '.stretching_max_error.dat'
      with open(name, mode) as f:
        f.write(str(stretching_max_error) + '\n')
    timer.timer('save_data')

    # Concatenate fibers
    xf = np.concatenate([fib_0.x, fib_1.x])

    # Create external density force
    # 1. Unbind microtubules
    timer.timer('fg_release')
    fg_0.release_particle(xf, np.array([num_points, num_points]), np.array([0, num_points]), dt)
    timer.timer('fg_release')
    # 2. Capture microtubules
    timer.timer('fg_trap')
    fg_0.trap_particle(xf, 2, np.array([num_points, num_points]), np.array([0, num_points]))
    timer.timer('fg_trap')
    # 3. Compute active and and passive force
    timer.timer('fg_force')
    force = fg_0.compute_force(xf, xs, 2, np.array([num_points, num_points]), np.array([0, num_points]))
    # force = np.zeros((num_points, 3))
    timer.timer('fg_force')
    # print('force = ', force.shape, '\n', force, '\n\n\n')
    
    # Set Boundary conditions
    # fib.set_BC(BC_end_0 = 'position', BC_end_vec_0 = np.array([5.0 + 3.0 * np.sin(step * dt), 0.5 * fib.length, 0.0]))

    # Get linear operator, RHS and apply BC
    timer.timer('A')
    A_0 = fib_0.form_linear_operator()
    A_1 = fib_1.form_linear_operator()
    timer.timer('A')
    timer.timer('RHS')
    RHS_0 = fib_0.compute_RHS(force_external = force[0:num_points, :])
    RHS_1 = fib_1.compute_RHS(force_external = force[num_points:2*num_points, :])
    timer.timer('RHS')
    timer.timer('BC')
    A_0, RHS_0 = fib_0.apply_BC(A_0, RHS_0)
    A_1, RHS_1 = fib_1.apply_BC(A_1, RHS_1)
    timer.timer('BC')

    # Solve linear system 
    timer.timer('solver')
    if method == 'dense_algebra':
      RHS = np.concatenate([RHS_0, RHS_1])
      Zero = np.zeros((4 * num_points, 4 * num_points))     
      A = np.vstack((np.hstack((A_0, Zero)),
                     np.hstack((Zero, A_1))))
      sol = np.linalg.solve(A, RHS) 
    elif method == 'iterative_block':
      counter = gmres_counter(print_residual = print_residual) 
      RHS = np.concatenate([RHS_0, RHS_1])
      Zero = np.zeros((4 * num_points, 4 * num_points))     
      A = np.vstack((np.hstack((A_0, Zero)),
                     np.hstack((Zero, A_1))))
      (LU_0, P_0) = scipy.linalg.lu_factor(A_0)
      (LU_1, P_1) = scipy.linalg.lu_factor(A_1)
      P = [P_0, P_1]
      LU = [LU_0, LU_1]
      def P_inv(LU, P, x):
        # return scipy.linalg.lu_solve((LU, P), x)
        y = np.empty_like(x)
        y[0:num_points*4] = scipy.linalg.lu_solve((LU[0], P[0]), x[0:num_points*4])
        y[num_points*4:num_points*8] = scipy.linalg.lu_solve((LU[1], P[1]), x[num_points*4:num_points*8])
        return y
      P_inv_partial = partial(P_inv, LU, P)
      P_inv_partial_LO = spla.LinearOperator((8*num_points, 8*num_points), matvec = P_inv_partial, dtype='float64')

      # Set collective BC
      if True:
        # Fix end of the fibers together
        I = np.eye(num_points)
        A[0,:]                                    =  0
        A[0,0:num_points]                         =  I[0,:] / dt
        A[0,4*num_points:5*num_points]            = -I[0,:] / dt
        A[num_points,:]                           =  0
        A[num_points,num_points:2*num_points]     =  I[0,:] / dt
        A[num_points,5*num_points:6*num_points]   = -I[0,:] / dt
        A[2*num_points,:]                         =  0
        A[2*num_points,2*num_points:3*num_points] =  I[0,:] / dt
        A[2*num_points,6*num_points:7*num_points] = -I[0,:] / dt

        A[4*num_points,:]                         =  0
        A[4*num_points,0:num_points]              = -I[0,:] / dt
        A[4*num_points,4*num_points:5*num_points] =  I[0,:] / dt
        A[5*num_points,:]                         =  0
        A[5*num_points,num_points:2*num_points]   = -I[0,:] / dt
        A[5*num_points,5*num_points:6*num_points] =  I[0,:] / dt
        A[6*num_points,:]                         =  0
        A[6*num_points,2*num_points:3*num_points] = -I[0,:] / dt
        A[6*num_points,6*num_points:7*num_points] =  I[0,:] / dt
        
        RHS[0]            = 0
        RHS[4*num_points] = 0
        RHS[num_points]   = 0
        RHS[5*num_points] = 0
        RHS[2*num_points] = 0
        RHS[6*num_points] = 0

      if True:
        # Fix angle between fibers
        I = np.eye(num_points)
        offset = 1
        A[offset,:]                                        = 0
        A[offset,0:num_points]                             =  fib_0.D_1[offset-1,:]
        A[offset,4*num_points:5*num_points]                = -fib_1.D_1[offset-1,:]
        A[offset+num_points,:]                             =  0
        A[offset+num_points,num_points:2*num_points]       =  fib_0.D_1[offset-1,:]
        A[offset+num_points,5*num_points:6*num_points]     = -fib_1.D_1[offset-1,:]
        A[offset+2*num_points,:]                           =  0
        A[offset+2*num_points,2*num_points:3*num_points]   =  fib_0.D_1[offset-1,:]  
        A[offset+2*num_points,6*num_points:7*num_points]   = -fib_1.D_1[offset-1,:]  

        A[offset + 4*num_points,:]                         =  0
        A[offset + 4*num_points,0:num_points]              = -fib_0.D_1[offset-1,:]
        A[offset + 4*num_points,4*num_points:5*num_points] =  fib_1.D_1[offset-1,:]
        A[offset + 5*num_points,:]                         =  0
        A[offset + 5*num_points,num_points:2*num_points]   = -fib_0.D_1[offset-1,:]
        A[offset + 5*num_points,5*num_points:6*num_points] =  fib_1.D_1[offset-1,:]
        A[offset + 6*num_points,:]                         =  0
        A[offset + 6*num_points,2*num_points:3*num_points] = -fib_0.D_1[offset-1,:]  
        A[offset + 6*num_points,6*num_points:7*num_points] =  fib_1.D_1[offset-1,:]  

        RHS[offset + 0]            =  angle_0_1[0]
        RHS[offset + 4*num_points] = -angle_0_1[0]
        RHS[offset + num_points]   =  angle_0_1[1]
        RHS[offset + 5*num_points] = -angle_0_1[1]
        RHS[offset + 2*num_points] =  angle_0_1[2]
        RHS[offset + 6*num_points] = -angle_0_1[2]

      # Transform to sparse matrix
      As = scipy.sparse.csr_matrix(A)
      (sol, info_precond) = spla.gmres(As, RHS, x0=x0, tol=tolerance, M=P_inv_partial_LO, maxiter=1000, restart=150, callback=counter) 
      x0 = sol   





    timer.timer('solver')      

    # Update fiber configuration
    fib_0.x[:,0] = sol[0:num_points]
    fib_0.x[:,1] = sol[num_points:2*num_points]
    fib_0.x[:,2] = sol[2*num_points:3*num_points]         
    fib_1.x[:,0] = sol[4*num_points:5*num_points]
    fib_1.x[:,1] = sol[5*num_points:6*num_points]
    fib_1.x[:,2] = sol[6*num_points:7*num_points]         
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
      f.write(str(num_points * 2) + '\n')
      y = np.empty((num_points * 2, 4))
      y[0:num_points, 0:3] = fib_0.x
      y[0:num_points,3] = sol[3*num_points:4*num_points]
      y[num_points:2*num_points, 0:3] = fib_1.x
      y[num_points:2*num_points,3] = sol[7*num_points:8*num_points]
      np.savetxt(f, y)
      print( y)

    xs = []
    xs.append( np.dot(fib_0.D_1, fib_0.x) )
    xs.append( np.dot(fib_1.D_1, fib_1.x) )
    xs = np.array(xs)
    xs = np.reshape(np.array(xs), (xs.size / 3, 3))
    stretching_error = np.sqrt(xs[:,0]**2 + xs[:,1]**2 + xs[:,2]**2) - 1.0
    stretching_max_error = max(stretching_error, key=abs)
    name = name_output + '.stretching_error.dat'
    with open(name, mode) as f:
      f.write(str(num_points * 2) + '\n')
      np.savetxt(f, stretching_error)
    name = name_output + '.stretching_max_error.dat'
    with open(name, mode) as f:
      f.write(str(stretching_max_error) + '\n')
  timer.timer('save_data')

  print( 'beta = %.4g' % fib_0.beta)
  print( 'beta = %.4g' % fib_1.beta)

  timer.timer(None, print_all = True)

  print( '# Main End')
