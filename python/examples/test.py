# Standard imports
from __future__ import print_function
import numpy as np
import scipy.sparse.linalg as spla
import scipy.linalg 
import scipy.optimize as sopt
from functools import partial
import sys

sys.path.append('../')



class gmres_counter(object):
  '''
  Callback generator to count iterations. 
  '''
  def __init__(self, print_residual = False):
    self.print_residual = print_residual
    self.niter = 0
  def __call__(self, x, rk=None):
    self.niter += 1
    if self.print_residual is True:
      if self.niter == 1:
        print( 'gmres =  0 1')
      print( 'gmres = ', self.niter, rk, '\n')









if __name__ == '__main__':
  print('Start')
  

  def nonlinear_function(x):
    xout = 2 * x[0] * x[1] + x[1]  + 1.0
    yout = -((x[0]**2 + x[1]**2) - 1.0)
    return np.array([xout, yout])

  def jacobian(x):
    J_11 = 2*x[1]
    J_12 = 2*x[0] + 1.0
    J_21 = -2 * x[0]
    J_22 = -2 * x[1]
    return np.vstack((np.hstack(J_11, J_12)), 
                     (np.hstack(J_21, J_22)))

  x = np.array([0., -1.00])

  # , jac = jacobian
  # sol_obj = sopt.root(nonlinear_function, x, method = 'krylov', tol = 1e-12, options = {'xtol' : 1e-12, 'maxiter' : 1000})
  counter = gmres_counter(print_residual = True)
  sol_obj = sopt.newton_krylov(nonlinear_function, x, method = 'gmres', f_rtol = 1e-10, verbose = True, callback = counter)
  print('sol_obj = \n', sol_obj)
  print('\n\n\nnonlinear_function = ', nonlinear_function(sol_obj))



  print( 'End')
