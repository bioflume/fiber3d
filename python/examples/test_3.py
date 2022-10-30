import numpy as np
from scipy.optimize import root
from scipy.sparse import spdiags, kron
from scipy.sparse.linalg import spilu, LinearOperator
import scipy
from numpy import cosh, zeros_like, mgrid, zeros, eye
from functools import partial
from scipy.optimize.nonlin import InverseJacobian, BroydenFirst, KrylovJacobian

# parameters
nx, ny = 50, 50
hx, hy = 1./(nx-1), 1./(ny-1)

P_left, P_right = 0, 0
P_top, P_bottom = 1, 0

def get_preconditioner():
    """Compute the preconditioner M"""
    diags_x = zeros((3, nx))
    diags_x[0,:] = 1/hx/hx
    diags_x[1,:] = -2/hx/hx
    diags_x[2,:] = 1/hx/hx
    Lx = spdiags(diags_x, [-1,0,1], nx, nx)

    diags_y = zeros((3, ny))
    diags_y[0,:] = 1/hy/hy
    diags_y[1,:] = -2/hy/hy
    diags_y[2,:] = 1/hy/hy
    Ly = spdiags(diags_y, [-1,0,1], ny, ny)

    J1 = kron(Lx, eye(ny)) + kron(eye(nx), Ly)

    # Now we have the matrix `J_1`. We need to find its inverse `M` --
    # however, since an approximate inverse is enough, we can use
    # the *incomplete LU* decomposition

    J1_ilu = spilu(J1)

    # This returns an object with a method .solve() that evaluates
    # the corresponding matrix-vector product. We need to wrap it into
    # a LinearOperator before it can be passed to the Krylov methods:

    M = LinearOperator(shape=(nx*ny, nx*ny), matvec=J1_ilu.solve)
    return M


class myPreconditioner(object):
    def __init__(self):
        self.J1_ilu = None
        return 
        
    def setup(self, P, f, func):
        self.update(P, f)
        return 

    def update(self, P, f):
        """Compute the preconditioner M"""
        diags_x = zeros((3, nx))
        diags_x[0,:] = 1/hx/hx
        diags_x[1,:] = -2/hx/hx
        diags_x[2,:] = 1/hx/hx
        Lx = spdiags(diags_x, [-1,0,1], nx, nx)
        
        diags_y = zeros((3, ny))
        diags_y[0,:] = 1/hy/hy
        diags_y[1,:] = -2/hy/hy
        diags_y[2,:] = 1/hy/hy
        Ly = spdiags(diags_y, [-1,0,1], ny, ny)
        
        J1 = kron(Lx, eye(ny)) + kron(eye(nx), Ly) 
        J1 = J1.todense() + 2.0 * np.diag(P)
        
        # Now we have the matrix `J_1`. We need to find its inverse `M` --
        # however, since an approximate inverse is enough, we can use
        # the *incomplete LU* decomposition
        
        J1_ilu = scipy.linalg.lu_factor(J1)
        self.J1_ilu = J1_ilu
        return 

    def solve(self, P, func=None, tol=0): 
        print 'solve xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' 
        return scipy.linalg.lu_solve(self.J1_ilu, P) 
        


def solve(preconditioning=True):
    """Compute the solution"""
    count = [0]

    def residual(P):
        count[0] += 1

        d2x = zeros_like(P)
        d2y = zeros_like(P)

        d2x[1:-1] = (P[2:]   - 2*P[1:-1] + P[:-2])/hx/hx
        d2x[0]    = (P[1]    - 2*P[0]    + P_left)/hx/hx
        d2x[-1]   = (P_right - 2*P[-1]   + P[-2])/hx/hx

        d2y[:,1:-1] = (P[:,2:] - 2*P[:,1:-1] + P[:,:-2])/hy/hy
        d2y[:,0]    = (P[:,1]  - 2*P[:,0]    + P_bottom)/hy/hy
        d2y[:,-1]   = (P_top   - 2*P[:,-1]   + P[:,-2])/hy/hy

        return d2x + d2y + P**2

    # preconditioner
    if preconditioning:
        M = get_preconditioner()
        # Pinv_partial = partial(preconditioner)
        # M = LinearOperator(shape=(nx*ny, nx*ny), matvec = Pinv_partial, dtype='float64')
        # M = InverseJacobian(myPreconditioner())
        # M = myPreconditioner()
        # M = InverseJacobian(myPreconditioner())
        # M = KrylovJacobian(inner_M=InverseJacobian(myPreconditioner()))

    else:
        M = None

    # solve
    guess = zeros((nx, ny), float)

    sol = root(residual, guess, method='krylov', tol=1e-10,
               options={'disp': True,
                        'jac_options': {'inner_M': M},
                        'method': 'gmres'})
    print('Residual', abs(residual(sol.x)).max())
    print('Evaluations', count[0])

    return sol.x

def main():
    sol = solve(preconditioning=False)
    print '\n=========================================== '
    print '=========================================== '
    print '=========================================== \n'
    sol = solve(preconditioning=True)



    # visualize
    import matplotlib.pyplot as plt
    x, y = mgrid[0:1:(nx*1j), 0:1:(ny*1j)]
    plt.clf()
    plt.pcolor(x, y, sol)
    plt.clim(0, 1)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()

