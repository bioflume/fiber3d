'''
Test to solve the 1D problem
D^2 P + P^2 = 0, with several boundary conditions.

dirichlet: P(x=0) = 1.0, P(x=1) = 0.
newmann: dP(x=0) = 1.0, dP(x=1) = 0.

cell_dirichlet:
(P_gl + P_0) / 2 = P_left
(P_{nx-1} + P_gr) / 2 = P_right

(d^2P)_0 = (P_gl - 2*P_0 + P_1) / h^2
         = (2*P_left - 3*P_0 + P_1) / h^2

cell_neumann:
P_gl = P_0 - h*dP_left
P_gr = P_{nx-1} + h*dP_right

(d^2P)_0 = (P_1 - 2*P_0 + P_gl) / h / h
         = (P_1 - P_0 - h*dP_left) / h / h
(d^2P)_{nx-1} = (P_gr - 2*P_{nx-1} + P_{nx-2}) / h / h
              = (h*dP_right - P_{nx-1} + P_{nx-2}) / h / h
'''
import numpy as np
from scipy.optimize import root
import scipy.sparse as scsp
import scipy.linalg as scla
import scipy.sparse.linalg as scspla
from functools import partial
import sys

sys.path.append('../')
from utils import nonlinear
from utils import cheb

# Centering = cell, nodes, chebyshev
mesh = 'chebyshev_dirichlet'

# Boundary conditions
P_left = -1.0
P_right = 0.
dP_left = 1.0
dP_right = 0.0

# Number of points in the mesh
nx = 32

# Define distance between points
if mesh == 'cell_dirichlet':
    hx = 1.0 / nx
    rx = np.linspace(0+hx/2.0, 1.0-hx/2.0, nx)
    n_free = nx
elif mesh == 'cell_neumann':
    hx = 1.0 / nx
    rx = np.linspace(0+hx/2.0, 1.0-hx/2.0, nx)
    n_free = nx
elif mesh == 'chebyshev_dirichlet':
    D_1, s = cheb.cheb(nx - 1)
    s = np.flipud(s)
    rx = (s + 1.0) / 2.0
    D_1 = np.flipud(np.flipud(D_1.T).T)
    D_1 = D_1 * (2.0 / 1.0)
    D_2 = np.dot(D_1, D_1)
    D_2 -= np.diag(np.sum(D_2.T, axis=0))
    n_free = nx + 2
elif mesh == 'chebyshev_neumann':
    D_1, s = cheb.cheb(nx - 1)
    s = np.flipud(s)
    rx = (s + 1.0) / 2.0
    D_1 = np.flipud(np.flipud(D_1.T).T)
    D_1 = D_1 * (2.0 / 1.0)
    D_2 = np.dot(D_1, D_1)
    D_2 -= np.diag(np.sum(D_2.T, axis=0))
    n_free = nx + 2

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


# Static Variable decorator for calculating acceptance rate.
def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate


@static_var('counter', 0)
def residual(P):
    '''
    D^2 P + P^2 = 0; P(x=0) = 1.0, P(x=1) = 0.
    '''
    residual.counter += 1
    if mesh == 'cell_dirichlet':
        dx2 = np.zeros_like(P)
        dx2[1:-1] = (P[2:] - 2*P[1:-1] + P[:-2]) / hx**2
        dx2[0] = (P[1] - 3*P[0] + 2*P_left) / hx**2
        dx2[-1] = (2*P_right - 3*P[-1] + P[-2]) / hx**2
        return dx2 + P**2
    elif mesh == 'cell_neumann':
        dx2 = np.zeros_like(P)
        dx2[1:-1] = (P[2:] - 2*P[1:-1] + P[:-2]) / hx**2
        dx2[0] = (P[1] - P[0] - hx * dP_left) / hx**2
        dx2[-1] = (hx * dP_right - P[-1] + P[-2]) / hx**2
        return dx2 + P**2
    elif mesh == 'chebyshev_dirichlet':
        res_0 = np.dot(D_2, P[0:-2]) + P[0:-2]**2
        res_0[0] += P[-2]
        res_0[-3] += P[-1]
        constraints = np.array([(P[0]-P_left), (P[-3]-P_right)])
        return np.concatenate([res_0, constraints])
    elif mesh == 'chebyshev_neumann':
        res_0 = np.dot(D_2, P[0:-2]) + P[0:-2]**2
        res_0[0] += P[-2]
        res_0[-3] += P[-1]
        constraints = np.array([np.dot(D_1[0], P[0:-2]) - dP_left, np.dot(D_1[-1], P[0:-2]) - dP_right])
        return np.concatenate([res_0, constraints])

    
def jacobian(P):
    if mesh == 'cell_dirichlet':
        J = np.zeros((nx,nx))

        diags_x = np.zeros((3, nx))
        diags_x[0,:] = 1/hx/hx
        diags_x[1,:] = -2/hx/hx
        diags_x[2,:] = 1/hx/hx

        diags_x[1,0] = -3/hx/hx
        diags_x[1,-1] = -3/hx/hx
        Lx = scsp.spdiags(diags_x, [-1,0,1], nx, nx)
        return Lx + np.diag(2.0 * P)
    elif mesh == 'cell_neumann':
        J = np.zeros((nx,nx))
        diags_x = np.zeros((3, nx))
        diags_x[0,:] = 1/hx/hx
        diags_x[1,:] = -2/hx/hx
        diags_x[2,:] = 1/hx/hx

        diags_x[1,0] = -1/hx/hx
        diags_x[1,-1] = -1/hx/hx
        Lx = scsp.spdiags(diags_x, [-1,0,1], nx, nx)
        return Lx + np.diag(2.0 * P)
    elif mesh == 'chebyshev_dirichlet':
        J = np.zeros((n_free, n_free))
        J[0:nx, 0:nx] = D_2 + np.diag(2.0 * P[0:-2])
        J[0, -2] = 1.0
        J[-3, -1] = 1.0
        J[-2,0] = 1.0
        J[-1,-3] = 1.0
        return J
    elif mesh == 'chebyshev_neumann':
        # res_0 = np.dot(D_2, P[0:-2]) + P[0:-2]**2
        # res_0[0] += P[-2]
        # res_0[-3] += P[-1]
        # constraints = np.array([np.dot(D_1[0], P[0:-2]) - dP_left, np.dot(D_1[-1], P[0:-2]) - dP_right])
        J = np.zeros((n_free, n_free))
        J[0:nx, 0:nx] = D_2 + np.diag(2.0 * P[0:-2])
        J[0, -2] = 1.0
        J[-3, -1] = 1.0
        J[-2,0:-2] = D_1[0]
        J[-1,0:-2] = D_1[-1]
        return J

def preconditioner(P,J):
    if mesh == 'cell_dirichlet':
        LU = scla.lu_factor(J)
        PC_partial = partial(scla.lu_solve, LU)
        return scspla.LinearOperator(J.shape, matvec=PC_partial)
    elif mesh == 'cell_neumann':
        Jinv = np.linalg.pinv(J)
        PC_partial = partial(np.dot, Jinv)
        return scspla.LinearOperator(J.shape, matvec=PC_partial)
    elif mesh == 'chebyshev_dirichlet':
        LU = scla.lu_factor(J)
        PC_partial = partial(scla.lu_solve, LU)
        return scspla.LinearOperator(J.shape, matvec=PC_partial)
    elif mesh == 'chebyshev_neumann':
        LU = scla.lu_factor(J)
        PC_partial = partial(scla.lu_solve, LU)
        return scspla.LinearOperator(J.shape, matvec=PC_partial)
    
# Set guess
guess = -0.1 * np.ones(n_free) 

# Solve with root
M = None
sol_root = root(residual, guess, method='krylov', tol=1e-6,
                options={'disp': True,
                         'jac_options': {'inner_M': M},
                         'method': 'gmres',
                         'maxiter': 1000})
print('Residual', abs(residual(sol_root.x)).max())
print('Evaluations', residual.counter)
print '==============================================================' 
print '\n\n\n'

# Solve with my solver
count = 0
if True:
    sol, res_norm, it, nonlinear_evaluations, gmres_iterations = nonlinear.nonlinear_solver(residual,
                                                                                            guess,
                                                                                            jacobian,
                                                                                            M=preconditioner,
                                                                                            tol=1e-6,
                                                                                            verbose=True,
                                                                                            max_outer_it=200)

    print '\n\n'
    print '============= Dirichlet ============'
    print 'constraints = ',  (sol[0]-P_left), (sol[-3]-P_right)
    print '============== Neumann ============='
    print 'constraints = ', np.dot(D_1[0], sol[0:-2]) - dP_left, np.dot(D_1[-1], sol[0:-2]) - dP_right
    print '\n\n'

else:
    sol = np.zeros_like(guess)

# Output data
rxP = np.concatenate([[rx],[sol_root.x[0:nx]], [sol[0:nx]]]).T
np.savetxt('data/test_1D.dat', rxP)



