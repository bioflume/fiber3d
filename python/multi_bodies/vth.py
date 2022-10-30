'''
This modules solves the mobility or the resistance problem for one
configuration of a multibody supensions and it can save some data like
the velocities or forces on the bodies, the mobility of a body or
the mobility of the blobs.
'''
from __future__ import division, print_function
import argparse
import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla
from scipy.spatial import ConvexHull
import subprocess
try:
  import cPickle as cpickle
except:
  try:
    import cpickle
  except:
    import _pickle as cpickle
from functools import partial, reduce
import sys
import time

# Find project functions
found_functions = False
path_to_append = ''
while found_functions is False:
  try:
    import multi_bodies
    from kernels import kernels
    from quaternion import quaternion
    from body import body 
    from read_input import read_input
    from read_input import read_vertex_file
    from read_input import read_clones_file
    from shape_gallery import shape_gallery
    from quadratures import Smooth_Closed_Surface_Quadrature_RBF
    from utils import miscellaneous
    from utils import gmres
    found_functions = True
  except ImportError:
    path_to_append += '../'
    print('searching functions in path ', path_to_append, sys.path)
    sys.path.append(path_to_append)
    if len(path_to_append) > 21:
      print('\nProjected functions not found. Edit path in multi_bodies_utilities.py')
      sys.exit()

print('searching functions in path ooo ', path_to_append, sys.path)

# Try to import the visit_writer (boost implementation)
# import visit.visit_writer as visit_writer
try:
  import visit.visit_writer as visit_writer
  # from visit import visit_writer
except ImportError:
  print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
  pass

from numba import njit, prange

# Callback generator
def make_callback():
  closure_variables = dict(counter=0, residuals=[]) 
  def callback(residuals):
    closure_variables["counter"] += 1
    closure_variables["residuals"].append(residuals)
    print(closure_variables["counter"], residuals)
  return callback



@njit(parallel=True)
def vth_numba(r_vec, R1, R2):
  v = np.zeros_like(r_vec)
  N = r_vec.size // 3
  r_vec = r_vec.reshape((N, 3))

  e = 4.0*R2**8 - 9.0*R1*R2**7 + 6.0*R1**3*R2**5 - 3.0*R1**5*R2**3 - 10.0*R1**6*R2**2 + 6.0*R1**8
  a = 3.0*R1*(R2**5 - R1**2*R2**3 + 2.0*R1**5) / e
  b = (-9.0*R1*R2**7 + 5.0*R1**3*R2**5 - 10.0*R1**6*R2**2) / e
  c = 3.0*R1*R2**8 / e
  d = (R2**3 - 2.0*R1**3)*R1**3*R2**5 / e

  print('a = ', a)
  print('b = ', b)
  print('c = ', c)
  print('d = ', d)
  print('e = ', e)

  
  for xn in prange(N):
    x = r_vec[xn, 0]
    y = r_vec[xn, 1]
    z = r_vec[xn, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    if r > 0:
      theta = np.arccos(z / r)
    else:
      theta = 0
    phi = np.arctan2(y, x)

    if r > R1 and r < R2:
      ur = np.cos(theta) * (1.0 - a*r**2 - b - 2*c/r + 2*d/r**3)
      uphi = 0.
      utheta = -np.sin(theta) * (1.0 - 2*a*r**2 - b - c/r - d/r**3)

      cos_theta = np.cos(theta)
      sin_theta = np.sin(theta)
      cos_phi = np.cos(phi)
      sin_phi = np.sin(phi)

      v[xn,0] = cos_phi*sin_theta*ur - sin_theta*uphi + cos_phi*cos_theta*utheta
      v[xn,1] = sin_phi*sin_theta*ur + cos_phi*uphi   + sin_phi*cos_theta*utheta
      v[xn,2] = (cos_theta*ur                         - sin_theta*utheta) 

  return v
    

@njit(parallel=True)
def vth_2_numba(r_vec, R1, R2):
  '''
  From Kallemov paper
  '''
  v = np.zeros_like(r_vec)
  N = r_vec.size // 3
  r_vec = r_vec.reshape((N, 3))

  l = R1 / R2
  alpha = 1 - 9*l/4 + 5*l**3/2 - 9*l**5/4 + l**6
  K = (1 - l**5) / alpha
  A = -3.75 * (l**3 - l**5) / alpha / R1**2
  B = 1.5 * R1 * (1-l**5) / alpha
  C = 0.5 * (1 + 5*l**3/4 - 9*l**5/4) / alpha
  D = 0.25 * R1**3 * (1 - l**3) / alpha
  
  for xn in prange(N):
    x = r_vec[xn, 0]
    y = r_vec[xn, 1]
    z = r_vec[xn, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    if r > 0:
      theta = np.arccos(z / r)
    else:
      theta = 0
    phi = np.arctan2(y, x)

    if r > R1 and r < R2:
      ur = -np.cos(theta) * (A*r**2/5 - B/r + 2*C + 2*D/r**3)
      uphi = 0.
      utheta = np.sin(theta) * (2*A*r**2/5 - 0.5*B/r + 2*C - D/r**3)

      cos_theta = np.cos(theta)
      sin_theta = np.sin(theta)
      cos_phi = np.cos(phi)
      sin_phi = np.sin(phi)

      v[xn,0] = cos_phi*sin_theta*ur - sin_theta*uphi + cos_phi*cos_theta*utheta
      v[xn,1] = sin_phi*sin_theta*ur + cos_phi  *uphi + sin_phi*cos_theta*utheta
      v[xn,2] = cos_theta        *ur                  - sin_theta        *utheta + 1.0

  return v
    

@njit(parallel=True)
def vth_inf_numba(r_vec, R1, R2):
  v = np.zeros_like(r_vec)
  N = r_vec.size // 3
  r_vec = r_vec.reshape((N, 3))

  c = 3.0*R1 / 4.0
  d = R1**3 / 4.0
  
  for xn in prange(N):
    x = r_vec[xn, 0]
    y = r_vec[xn, 1]
    z = r_vec[xn, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    if r > 0:
      theta = np.arccos(z / r)
    else:
      theta = 0
    phi = np.arctan2(y, x)

    if r < R1:
      ur = 0
      utheta = 0
    elif r > R2:
      ur = 0
      utheta = 0
    else:
      ur = np.cos(theta) * (1.0 - 2*c/r + 2*d/r**3)
      uphi = 0.0
      utheta = -np.sin(theta) * (1.0 - c/r - d/r**3)

    if r > R1 and r < R2:
      cos_theta = np.cos(theta)
      sin_theta = np.sin(theta)
      cos_phi = np.cos(phi)
      sin_phi = np.sin(phi)

      v[xn,0] = cos_phi*sin_theta*ur - sin_theta*uphi + cos_phi*cos_theta*utheta
      v[xn,1] = sin_phi*sin_theta*ur + cos_phi*uphi   + sin_phi*cos_theta*utheta
      v[xn,2] = (cos_theta*ur                         - sin_theta*utheta) - 1.0

  return v






if __name__ == '__main__':
  '''
  This function plots the velocity field to a grid. 
  '''
  # Prepare grid values
  output = 'data/convergence/run.theory'
  grid = np.array([-2., 2., 400, 0., 0., 1, -2., 2., 400])
  grid = np.reshape(grid, (3,3)).T
  grid_length = grid[1] - grid[0]
  grid_points = np.array(grid[2], dtype=np.int32)
  num_points = reduce(lambda x,y: x*y, grid_points)

  # Set grid coordinates
  dx_grid = grid_length / grid_points
  grid_x = np.array([grid[0,0] + dx_grid[0] * (x+0.5) for x in range(grid_points[0])])
  grid_y = np.array([grid[0,1] + dx_grid[1] * (x+0.5) for x in range(grid_points[1])])
  grid_z = np.array([grid[0,2] + dx_grid[2] * (x+0.5) for x in range(grid_points[2])])
  # Be aware, x is the fast axis.
  zz, yy, xx = np.meshgrid(grid_z, grid_y, grid_x, indexing = 'ij')
  grid_coor = np.zeros((num_points, 3))
  grid_coor[:,0] = np.reshape(xx, xx.size)
  grid_coor[:,1] = np.reshape(yy, yy.size)
  grid_coor[:,2] = np.reshape(zz, zz.size)


  grid_velocity = vth_2_numba(grid_coor, 0.2, 2.0) # + vth_numba(grid_coor, 1.0, 2.0)


  
  
  # Prepara data for VTK writer 
  variables = [np.reshape(grid_velocity, grid_velocity.size)] 
  dims = np.array([grid_points[0]+1, grid_points[1]+1, grid_points[2]+1], dtype=np.int32) 
  nvars = 1
  vardims = np.array([3])
  centering = np.array([0])
  varnames = ['velocity\0']
  name = output + '.velocity_field.vtk'
  grid_x = grid_x - dx_grid[0] * 0.5
  grid_y = grid_y - dx_grid[1] * 0.5
  grid_z = grid_z - dx_grid[2] * 0.5
  grid_x = np.concatenate([grid_x, [grid[1,0]]])
  grid_y = np.concatenate([grid_y, [grid[1,1]]])
  grid_z = np.concatenate([grid_z, [grid[1,2]]])

  print('dims = ', dims)
  print('\n\n\n')

  # Write velocity field
  visit_writer.boost_write_rectilinear_mesh(name,      # File's name
                                            0,         # 0=ASCII,  1=Binary
                                            dims,      # {mx, my, mz}
                                            grid_x,     # xmesh
                                            grid_y,     # ymesh
                                            grid_z,     # zmesh
                                            nvars,     # Number of variables
                                            vardims,   # Size of each variable, 1=scalar, velocity=3*scalars
                                            centering, # Write to cell centers of corners
                                            varnames,  # Variables' names
                                            variables) # Variables
  
