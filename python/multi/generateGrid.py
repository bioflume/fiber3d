'''
Solve the linear system M_SB * lambda_B = v_S

where v_S is defined in a surface surrounding swimmer B
and lambda_B is defined on the blobs of swimmer B.

v_S can be computed like v_S = M_SA * lambda_A
using the force on the blobs of swimmer A.
'''
import numpy as np
from functools import partial
from decimal import Decimal
import sys
 

def gauss_weights(N):
  '''
  Compute Legendre points and weights for Gauss quadrature.
  From Spectral Methods in MATLAB, Trefethen 2000, Program Gauss.m (page 129).
  '''
  s = np.arange(1, N)
  beta = 0.5 / np.sqrt(1.0 - 1.0 / (2 * s)**2)
  T = np.diag(beta, k=1) + np.diag(beta, k=-1)
  eig_values, eig_vectors = np.linalg.eigh(T)
  w = 2 * eig_vectors[0,:]**2
  return eig_values, w


def parametrization(p):
  '''
  Set parametrization, (u,v), with p+1 points along u and (2*p+2) along v.
  In total 2*p**2 points because at the poles we only have one point.

  Return parametrization and weights.
  '''
  # Precomputation
  Nu = p + 1
  Nv = 2 * (p + 1)
  N = Nu * Nv
  t, w_gauss = gauss_weights(Nu)
  u = np.arccos(t)
  v = np.linspace(0, 2*np.pi, Nv, endpoint=False)
  uu, vv = np.meshgrid(u, v, indexing = 'ij')

  # Parametrization
  uv = np.zeros((N,2))
  uv[:,0] = uu.flatten()
  uv[:,1] = vv.flatten()

  # Weights precomputation
  uw = w_gauss / np.sin(u)
  vw = np.ones(v.size) * 2 * np.pi / Nv  
  uuw, vvw = np.meshgrid(uw, vw, indexing = 'ij')
  
  # Weights
  w = uuw.flatten() * vvw.flatten()
  
  return uv, w


def sphere(a, uv):
  '''
  Return the points on a sphere of radius a parametrized by uv.
  '''
  # Generate coordinates
  x = np.zeros((uv.shape[0], 3))
  x[:,0] = a * np.cos(uv[:,1]) * np.sin(uv[:,0])
  x[:,1] = a * np.sin(uv[:,1]) * np.sin(uv[:,0])
  x[:,2] = a * np.cos(uv[:,0]) 
  return x



def generate_grid(p,radius):
  uv, w = parametrization(p)
  x = sphere(radius, uv)

  return x