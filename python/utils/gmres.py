'''
Wrapper for scipy gmres to use right preconditioner.
'''

import numpy as np
import scipy.sparse.linalg as scspla
from functools import partial

def gmres(A, b, x0=None, tol=1e-05, restart=None, maxiter=None, xtype=None, M=None, callback=None, restrt=None, PC_side='right'):
  '''
  Solve the linear system A*x = b, using right or left preconditioning.
  Inputs and outputs as in scipy gmres plus PC_side ('right' or 'left').

  Right Preconditioner (default):
    First solve A*P^{-1} * y = b for y
    then solve P*x = y, for x.

  Left Preconditioner;
    Solve P^{-1}*A*x = P^{-1}*b


  Use Generalized Minimal Residual to solve A x = b.

  Parameters
  ----------
  A : {sparse matrix, dense matrix, LinearOperator}
      Matrix that defines the linear system.
  b : {array, matrix}
      Right hand side of the linear system. It can be a matrix.

  Returns
  -------
  x : {array, matrix}
      The solution of the linear system.
  info : int
      Provides convergence information:
        * 0  : success
        * >0 : convergence to tolerance not achieved, number of iterations
        * <0 : illegal input or breakdown

  Other parameters
  ----------------
  PC_side: {'right', 'left'}
      Use right or left Preconditioner. Right preconditioner (default) uses
      the real residual to determine convergence. Left preconditioner uses
      a preconditioned residual (M*r^n = M*(b - A*x^n)) to determine convergence.
  x0 : {array, matrix}
      Initial guess for the linear system (zero by default).
  tol : float
      Tolerance. The solver finishes when the relative or the absolute residual  
      norm are below this tolerance.
  restart : int, optional
      Number of iterations between restarts. 
      Default is 20.
  maxiter : int, optional
      Maximum number of iterations.  
  xtype : {'f','d','F','D'}
      This parameter is DEPRECATED --- avoid using it.
      The type of the result.  If None, then it will be determined from
      A.dtype.char and b.  If A does not have a typecode method then it
      will compute A.matvec(x0) to get a typecode.   To save the extra
      computation when A does not have a typecode attribute use xtype=0
      for the same type as b or use xtype='f','d','F',or 'D'.
      This parameter has been superseded by LinearOperator.
  M : {sparse matrix, dense matrix, LinearOperator}
      Inverse of the preconditioner of A. By default M is None.
  callback : function
      User-supplied function to call after each iteration.  It is called
      as callback(rk), where rk is the current residual vector.
  restrt : int, optional
      DEPRECATED - use `restart` instead.

  See Also
  --------
  LinearOperator

  Notes
  -----
  A preconditioner, P, is chosen such that P is close to A but easy to solve
  for. The preconditioner parameter required by this routine is
  ``M = P^-1``. The inverse should preferably not be calculated
  explicitly.  Rather, use the following template to produce M::

  # Construct a linear operator that computes P^-1 * x.
  import scipy.sparse.linalg as spla
  M_x = lambda x: spla.spsolve(P, x)
  M = spla.LinearOperator((n, n), M_x)
  '''

  # If left preconditioner (or no Preconditioner) just call scipy gmres
  if PC_side == 'left' or M is None:
    return scspla.gmres(A, b, M=M, x0=x0, tol=tol, maxiter=maxiter, restart=restart, callback=callback)

  # Create LinearOperator for A and P^{-1}
  A_LO = scspla.aslinearoperator(A)
  M_LO = scspla.aslinearoperator(M)

  # Define new LinearOperator A*P^{-1}
  def APinv(x,A,M):
    return A.matvec(M_LO.matvec(x))
  APinv_partial = partial(APinv, A=A_LO, M=M_LO)
  APinv_partial_LO = scspla.LinearOperator((b.size, b.size), matvec = APinv_partial, dtype='float64') 

  # Solve system A*P^{-1} * y = b
  (y, info) = scspla.gmres(APinv_partial_LO, b, x0=x0, tol=tol, maxiter=maxiter, restart=restart, callback=callback) 

  # Solve system P*x = y
  x = M_LO.matvec(y)
  
  # Return solution and info
  return x, info



