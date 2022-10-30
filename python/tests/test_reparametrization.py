import numpy as np
from functools import partial
import sys
sys.path.append('../')
from utils import cheb
from utils import nonlinear


if __name__ == '__main__':
  print('# Start')
  # Set some parameters
  N = 16
  L = 2.0 * np.pi
  R = 2.0 * np.pi / L

  # Create cheb points 
  D_1, alpha = cheb.cheb(N - 1)
  alpha = np.flipud(alpha)
  s = alpha
  D_1 = np.flipud(np.flipud(D_1.T).T)
  D_1 *= (2.0 / L)

  # Create points with wrong inextensibility;
  # initial configuration is close to a circle
  x = np.zeros((N, 3))
  x[:,0] = R * np.cos(np.pi * s) + np.random.randn(N) * 0.001
  x[:,1] = R * np.sin(np.pi * s)
  
  # Define function to minimize
  def func(s_interior, cheb_coeff_x, cheb_coeff_y, cheb_coeff_z, D_1):
    s = np.zeros(s_interior.size + 2)
    s[1:-1] = s_interior
    s[0] = -1.0
    s[-1] = 1.0
    x = cheb.cheb_eval(s, cheb_coeff_x, 1)
    y = cheb.cheb_eval(s, cheb_coeff_y, 1)
    z = cheb.cheb_eval(s, cheb_coeff_z, 1)
    
    xs = np.dot(D_1, x)
    ys = np.dot(D_1, y)
    zs = np.dot(D_1, z)
    residual = (xs*xs + ys*ys + zs*zs - 1.0)
    return residual[1:-1]

  # Calc coefficients
  cheb_coeff_x = cheb.cheb_calc_coef(x[:,0])
  cheb_coeff_y = cheb.cheb_calc_coef(x[:,1])
  cheb_coeff_z = cheb.cheb_calc_coef(x[:,2])

  # Prepare function
  func_partial = partial(func,
                         cheb_coeff_x=cheb_coeff_x,
                         cheb_coeff_y=cheb_coeff_y,
                         cheb_coeff_z=cheb_coeff_z,
                         D_1=D_1)

  # Get interior points
  s_interior = s[1:-1]

  # Move interior points
  print('s = \n', s)
  print('\n\n\n')
  s_interior, res_norm, iterations, nonlinear_evaluations, gmres_iterations = nonlinear.nonlinear_solver(func_partial,
                                                                                                         s_interior,
                                                                                                         jacobian=None,
                                                                                                         tol=1e-6,
                                                                                                         M=None,
                                                                                                         max_outer_it=10,
                                                                                                         max_inner_it=1000,
                                                                                                         gmres_restart=1000,
                                                                                                         verbose=True,
                                                                                                         gmres_verbose=True)
  s_new = np.zeros(N)
  s[0] = -1.0
  s[1:-1] = s_interior
  s[-1] = 1.0
  
  print('res_norm              = ', res_norm)
  print('iterations            = ', iterations)
  print('nonlinear_evaluations = ', nonlinear_evaluations)
  print('gmres_iterations      = ', gmres_iterations)

  print('\n\n')
  print('s = \n', s)
  print('\n\n')
