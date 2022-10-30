'''
Small utility functions.
'''
from __future__ import division, print_function
import numpy as np



# Static Variable decorator 
def static_var(varname, value):
  def decorate(func):
    setattr(func, varname, value)
    return func
  return decorate


def find_extrema():
  '''
  Return the maximum and minimum value of each dimension
  of an array.
  '''
  return None


def crossProductMatrix(x):
  '''
  Return the 3x3 matrix such as 
  x \cross b = Xb
  
  for any vector b.
  '''
  X = np.zeros((3,3))
  X[0,1] = -x[2]
  X[0,2] =  x[1]
  X[1,0] =  x[2]
  X[1,2] = -x[0]
  X[2,0] = -x[1]
  X[2,1] =  x[0]
  return X


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
        print('gmres =  0 1')
      print('gmres = ', self.niter, rk)

