'''
Module with a simple nonlinear solver.
'''
from __future__ import print_function
import numpy as np
import scipy.sparse.linalg as scspla
import sys
from functools import partial

def nonlinear_solver(F, x, jacobian=None, tol=1e-12, M=None, max_outer_it=1000, max_inner_it=1000, gmres_restart=1000, verbose=False, gmres_verbose=False):
  '''
  Newton-Krylov method to solve a linear solver.
  
  For each Newton iteration the solver does:
  1. Initialize Jacobian matrix vector product at x_n.
  2. Initialize Preconditioner for linear system.
  3. Solve linear system for dx: Jacobian(x_n) dx = -F(x_n)
  4. Linear search for alpha, such as x_{n+1} = x_n + alpha * dx has a smaller residual than x_n.
  5. Compute the residual at x_n and decide if the desired tolerance has been attained.
  
  Inputs:
  F = nonlinear function to evaluate the residual.
  x = solution initial guess.
  Jacobian = function to compute the product with the Jacobian. It should return a Linear Operator or a matrix.
  tol = tolerance for the nonlinear solver.
  M = function to compute the action of the preconditioner inverse for the inner linear system; 
      i.e. M(x_n)*J(x_n) \approx Identity. It should return a LinearOperator or a matrix.
  max_outer_i = maximum number of iterations for the outer solver.
  max_inner_i = maximum number of iterations for each inner solver.
  gmres_restart = iteration number to restart GMRES.
  verbose = print information about convergence.
  gmres_verbose = print information about GMRES convergence.

  Outputs:
  x = solution of the nonlinear system.
  res_norm = residual norm of the solution.
  iterations = number of Newton iterations.
  nonlinear_evaluations = number of nonlinear function evaluations.
  gmres_iterations = total number of gmres iterations.
  '''
  # Init counters
  gmres_iterations = 0
  nonlinear_evaluations = 0

  # Define Jacobian product with finite differences
  order_finite_differences = 2
  def J_product(x, F, res, x_n, order):
    epsilon_J = np.sqrt(1.0e-06 * (1.0 + np.linalg.norm(x_n))) / np.linalg.norm(x)
    if order == 1:
      return (F(x_n + epsilon_J * x) - res) / epsilon_J
    elif order == 2:
      return (F(x_n + epsilon_J * x) - F(x_n - epsilon_J * x)) / (2.0 * epsilon_J)

  # Linear solver tolerance selection, copied from scipy nonlin.py
  gamma = 0.9
  eta_max = 0.9999
  eta_treshold = 0.1
  eta = 1e-3
  
  # Compute initial residual
  res = F(x)
  res_norm = np.linalg.norm(res, ord=np.inf)
  nonlinear_evaluations += 1
  
  for it in range(max_outer_it):
    if verbose:
      print( 'outer it = ', it, '; nonlinear evaluations = ', nonlinear_evaluations, 
             '; total gmres it = ', gmres_iterations, '; gmres_tol = ', eta, '; residual = ', res_norm)
        
    # Break loop if tolerance has been attained
    if res_norm <= tol:
      break

    # Build Jacobian and Preconditioner inverse
    if jacobian is None:
      J_product_partial = partial(J_product,
                                  F = F, 
                                  res = res, 
                                  x_n = x, 
                                  order = order_finite_differences)
      J = scspla.LinearOperator((res.size, res.size), matvec = J_product_partial, dtype='float64')      
      PC = None if M is None else M(x, None)
    else:
      J = jacobian(x)
      PC = None if M is None else M(x, J)

    # Scale RHS to norm 1
    if res_norm > 0:
      res = res / res_norm

    counter = gmres_counter(print_residual = gmres_verbose)
    (sol_precond, info_precond) = scspla.gmres(J, -1.0 * res.flatten(), tol=eta, M=PC, maxiter=max_inner_it, restart=gmres_restart, callback=counter) 
    if info_precond != 0:
      raise Exception('GMRES didn\'t converge. It returned info = ', info_precond)
 
    # Scale solution with RHS norm
    if res_norm > 0:
      sol_precond = sol_precond * res_norm

    # Update iterations
    gmres_iterations += counter.niter
    if jacobian is None:
      nonlinear_evaluations += counter.niter * order_finite_differences

    # Compute new solution with simple line search 
    line_factor = 1.0
    while True:
      x_new = x + line_factor * np.reshape(sol_precond, x.shape)

      # Compute new residual
      res = F(x_new)
      res_norm_new = np.linalg.norm(res, ord=np.inf)
      nonlinear_evaluations += 1
        
      # If residual doesn't decrease enough use line search
      if (res_norm_new < (1 - 0.25 * line_factor) * res_norm) or line_factor < min(tol, 1e-15):
        x = x_new
        # print('x = \n', x)
        break

      if verbose:
        print( 'line search, res = ', res_norm_new, 'line_factor = ', line_factor)
        pass 

      line_factor = line_factor * 0.25

    # Adjust forcing parameters for inexact methods, copied from scipy nonlin.py 
    eta_A = gamma * res_norm_new**2 / res_norm**2
    if gamma * eta**2 < eta_treshold:
      eta = min(eta_max, eta_A)
    else:
      eta = min(eta_max, max(eta_A, gamma*eta**2))
    res_norm = res_norm_new

  if it >= max_outer_it - 1:
    it += 1
    # raise Exception('Outer solver didn\'t converge. res_norm = ', res_norm, '; it = ', it, '; nonlinear_evaluations = ', nonlinear_evaluations, '; gmres_iterations = ', gmres_iterations)
    print('solver didn\'t converge')
       
  return (x, res_norm, it, nonlinear_evaluations, gmres_iterations)



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
      
