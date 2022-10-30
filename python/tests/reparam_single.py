from __future__ import print_function
import sys
sys.path.append('../../')

import numpy as np
from utils import cheb
from fiber import fiber
from utils import barycentricMatrix as bary
import scipy.io 

mat = scipy.io.loadmat('bendFiber.mat')

X = mat['X']

# Get the matrices
num_points = 32
num_points_maxUp = 96

D_1, alpha = cheb.cheb(num_points - 1)
alpha = np.flipud(alpha)

D_1_maxUp, alpha_maxUp = cheb.cheb(num_points_maxUp - 1)
alpha_maxUp = np.flipud(alpha_maxUp)
P_maxUp = bary.barycentricMatrix(alpha, alpha_maxUp)

D_1_0 = np.flipud(np.flipud(D_1.T).T)
D_1_0_maxUp = np.flipud(np.flipud(D_1_maxUp.T).T)



# Fiber configurations before reparametrization

xup = np.dot(P_maxUp,X)
x_a = np.dot(D_1_0_maxUp,xup)
integrand = np.sqrt(x_a[:,0]**2 + x_a[:,1]**2 + x_a[:,2]**2)
length = np.trapz(integrand,alpha_maxUp)

fib = fiber.fiber(num_points = num_points, 
		dt = 5e-3,
		E = 0.1,
		length = length,
		epsilon = 1e-3,
		num_points_finite_diff = 0)
fib.x = X
fib.find_upsample_rate()


# Now, reparameterize
Xn2, Xn3, Xn4, Xn5 = np.zeros((num_points,3)), np.zeros((num_points,3)), np.zeros((num_points,3)), np.zeros((num_points,3))
coeffXn2, coeffXn3, coeffXn4, coeffXn5 = np.zeros((num_points,3)), np.zeros((num_points,3)),np.zeros((num_points,3)), np.zeros((num_points,3)) 
quality_meas = np.zeros(4)
coeffX = np.zeros((num_points,3)) 
quality_meas_0 = np.zeros(4)
modes = np.linspace(0,num_points-1,num_points)
coeffX[:,0] = cheb.cheb_calc_coef(fib.x[:,0]) 
coeffX[:,1] = cheb.cheb_calc_coef(fib.x[:,1])
coeffX[:,2] = cheb.cheb_calc_coef(fib.x[:,2])

# Keep the original x0, assign it after every reparametrization
x0 = fib.x
  
niter = fib.reparameterize(30,4)
#fib.correct()
print('Took ', niter, ' iterations')
Xn2 = fib.x
coeffXn2[:,0] = cheb.cheb_calc_coef(fib.x[:,0])
coeffXn2[:,1] = cheb.cheb_calc_coef(fib.x[:,1])
coeffXn2[:,2] = cheb.cheb_calc_coef(fib.x[:,2])
fib.x = x0
#quality_meas[0,i] = np.sum(modes**2 * (coeffXn2[:,0,i]**2 + coeffXn2[:,1,i]**2 + coeffXn2[:,2,i]**2))

niter = fib.reparameterize(30,5)
#fib.correct()
print('Took ', niter, ' iterations')
Xn3 = fib.x
coeffXn3[:,0] = cheb.cheb_calc_coef(fib.x[:,0])
coeffXn3[:,1] = cheb.cheb_calc_coef(fib.x[:,1])
coeffXn3[:,2] = cheb.cheb_calc_coef(fib.x[:,2])
fib.x = x0
#quality_meas[1,i] = np.sum(modes**3 * (coeffXn3[:,0,i]**2 + coeffXn3[:,1,i]**2 + coeffXn3[:,2,i]**2))
  
niter = fib.reparameterize(30,6)
#fib.correct()
print('Took ', niter, ' iterations')
Xn4 = fib.x
coeffXn4[:,0] = cheb.cheb_calc_coef(fib.x[:,0])
coeffXn4[:,1] = cheb.cheb_calc_coef(fib.x[:,1])
coeffXn4[:,2] = cheb.cheb_calc_coef(fib.x[:,2])
fib.x = x0
  
#quality_meas[2,i] = np.sum(modes**4 * (coeffXn4[:,0,i]**2 + coeffXn4[:,1,i]**2 + coeffXn4[:,2,i]**2))

niter = fib.reparameterize(30,7)
print('Took ', niter, ' iterations')
#fib.correct()
Xn5 = fib.x
coeffXn5[:,0] = cheb.cheb_calc_coef(fib.x[:,0])
coeffXn5[:,1] = cheb.cheb_calc_coef(fib.x[:,1])
coeffXn5[:,2] = cheb.cheb_calc_coef(fib.x[:,2])
#quality_meas[3] = np.sum(modes**5 * (coeffXn5[:,0,i]**2 + coeffXn5[:,1,i]**2 + coeffXn5[:,2,i]**2))
fib.x = x0

	
# Prepare output
mat['Xn2'] = Xn2
mat['Xn3'] = Xn3
mat['Xn4'] = Xn4
mat['Xn5'] = Xn5
mat['coeffXn2'] = coeffXn2
mat['coeffXn3'] = coeffXn3
mat['coeffXn4'] = coeffXn4
mat['coeffXn5'] = coeffXn5
mat['coeffX'] = coeffX
scipy.io.savemat('reparamBendFib_30iters.mat',mat)

#mat['quality_meas'] = quality_meas

