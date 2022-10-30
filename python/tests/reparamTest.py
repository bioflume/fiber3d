from __future__ import print_function
import sys
sys.path.append('../../')

import numpy as np
from utils import cheb
from fiber import fiber
from utils import barycentricMatrix as bary
import scipy.io 

mat = scipy.io.loadmat('fibConfigsForReparam.mat')
xbef = mat['x2rep']
ybef = mat['y2rep']
zbef = mat['z2rep']

xaft = mat['xAFT']
yaft = mat['yAFT']
zaft = mat['zAFT']

# Get the matrices
num_points = 8
num_points_maxUp = 96

D_1, alpha = cheb.cheb(num_points - 1)
alpha = np.flipud(alpha)

num_points = 8
D_1_up, alpha_up = cheb.cheb(num_points-1)
alpha_up = np.flipud(alpha_up)
P_up = bary.barycentricMatrix(alpha, alpha_up)

#D_1, alpha = np.copy(D_1_up), np.copy(alpha_up)

D_1_maxUp, alpha_maxUp = cheb.cheb(num_points_maxUp - 1)
alpha_maxUp = np.flipud(alpha_maxUp)
P_maxUp = bary.barycentricMatrix(alpha, alpha_maxUp)

D_1_0 = np.flipud(np.flipud(D_1.T).T)
D_1_0_maxUp = np.flipud(np.flipud(D_1_maxUp.T).T)



# Fiber configurations before reparametrization
fibers = []
x2rep = np.zeros((num_points,6))
y2rep = np.zeros((num_points,6))
z2rep = np.zeros((num_points,6))
for i in range(6):
  X, Xaft = np.zeros((8,3)), np.zeros((8,3))
  X[:,0] = xbef[:,i]
  X[:,1] = ybef[:,i]
  X[:,2] = zbef[:,i]
  #X = np.dot(P_up, X)

  x2rep[:,i] = X[:,0]
  y2rep[:,i] = X[:,1]
  z2rep[:,i] = X[:,2]
  
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

  fib.set_BC(BC_start_0='velocity', BC_start_1='angular_velocity')

  fibers.append(fib)

# Now, reparameterize
Xn2, Xn3, Xn4, Xn5 = np.zeros((num_points,3,6)), np.zeros((num_points,3,6)), np.zeros((num_points,3,6)), np.zeros((num_points,3,6))
coeffXn2, coeffXn3, coeffXn4, coeffXn5 = np.zeros((num_points,3,6)), np.zeros((num_points,3,6)),np.zeros((num_points,3,6)), np.zeros((num_points,3,6)) 
quality_meas = np.zeros((4,6))
coeffX = np.zeros((num_points,3,6)) 
quality_meas_0 = np.zeros((4,6))
i = 0
modes = np.linspace(0,num_points-1,num_points)
for fib in fibers:
  coeffX[:,0,i] = cheb.cheb_calc_coef(fib.x[:,0])
  coeffX[:,1,i] = cheb.cheb_calc_coef(fib.x[:,1])
  coeffX[:,2,i] = cheb.cheb_calc_coef(fib.x[:,2])

  # Keep the original x0, assign it after every reparametrization
  x0 = fib.x
  
  niter = fib.reparameterize(300,4)
  #fib.correct()
  print('Took ', niter, ' iterations')
  Xn2[:,:,i] = fib.x
  coeffXn2[:,0,i] = cheb.cheb_calc_coef(fib.x[:,0])
  coeffXn2[:,1,i] = cheb.cheb_calc_coef(fib.x[:,1])
  coeffXn2[:,2,i] = cheb.cheb_calc_coef(fib.x[:,2])
  fib.x = x0
  #quality_meas[0,i] = np.sum(modes**2 * (coeffXn2[:,0,i]**2 + coeffXn2[:,1,i]**2 + coeffXn2[:,2,i]**2))

  niter = fib.reparameterize(300,5)
  #fib.correct()
  print('Took ', niter, ' iterations')
  Xn3[:,:,i] = fib.x
  coeffXn3[:,0,i] = cheb.cheb_calc_coef(fib.x[:,0])
  coeffXn3[:,1,i] = cheb.cheb_calc_coef(fib.x[:,1])
  coeffXn3[:,2,i] = cheb.cheb_calc_coef(fib.x[:,2])
  fib.x = x0
  #quality_meas[1,i] = np.sum(modes**3 * (coeffXn3[:,0,i]**2 + coeffXn3[:,1,i]**2 + coeffXn3[:,2,i]**2))
  
  niter = fib.reparameterize(300,6)
  #fib.correct()
  print('Took ', niter, ' iterations')
  Xn4[:,:,i] = fib.x
  coeffXn4[:,0,i] = cheb.cheb_calc_coef(fib.x[:,0])
  coeffXn4[:,1,i] = cheb.cheb_calc_coef(fib.x[:,1])
  coeffXn4[:,2,i] = cheb.cheb_calc_coef(fib.x[:,2])
  fib.x = x0
  
  #quality_meas[2,i] = np.sum(modes**4 * (coeffXn4[:,0,i]**2 + coeffXn4[:,1,i]**2 + coeffXn4[:,2,i]**2))

  niter = fib.reparameterize(300,7)
  print('Took ', niter, ' iterations')
  #fib.correct()
  Xn5[:,:,i] = fib.x
  coeffXn5[:,0,i] = cheb.cheb_calc_coef(fib.x[:,0])
  coeffXn5[:,1,i] = cheb.cheb_calc_coef(fib.x[:,1])
  coeffXn5[:,2,i] = cheb.cheb_calc_coef(fib.x[:,2])
  quality_meas[3,i] = np.sum(modes**5 * (coeffXn5[:,0,i]**2 + coeffXn5[:,1,i]**2 + coeffXn5[:,2,i]**2))
  fib.x = x0
  i += 1

	
# Prepare output
mat['x2rep'] = x2rep
mat['y2rep'] = y2rep
mat['z2rep'] = z2rep
mat['Xn2'] = Xn2
mat['Xn3'] = Xn3
mat['Xn4'] = Xn4
mat['Xn5'] = Xn5
mat['coeffXn2'] = coeffXn2
mat['coeffXn3'] = coeffXn3
mat['coeffXn4'] = coeffXn4
mat['coeffXn5'] = coeffXn5
mat['coeffX'] = coeffX
scipy.io.savemat('reparamTestResults_4567_300iters_HALFmodes.mat',mat)

#mat['quality_meas'] = quality_meas

