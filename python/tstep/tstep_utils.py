from __future__ import division, print_function
import numpy as np
import imp
import sys
import time
import argparse
import subprocess
import scipy.sparse as scsp
import scipy.linalg as scla
import scipy.sparse.linalg as scspla
sys.path.append('../')
sys.path.append('./')
try:
  import cPickle as cpickle
except:
  try:
    import cpickle
  except:
    import _pickle as cpickle

from utils import timer
from utils import cheb
from utils import barycentricMatrix as bary
import scipy.integrate as spi
from utils import finite_diff
from fiber import fiber
from kernels import kernels
from utils import miscellaneous
from body import body
from molecular_motor import molecular_motor

try:
  #from numba import njit, prange
  from numba.typed import List
except ImportError:
  print('Numba not found')

# If pycuda is installed import kernels_pycuda
try: 
  imp.find_module('pycuda')
  found_pycuda = True
except ImportError:
  found_pycuda = False

from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

##############################################################################################
def distribute_equally_spaced(fibers):
  '''
  '''
  uprate = 12
  for k, fib in enumerate(fibers):
    num_points = fib.num_points
    num_points_up = uprate * num_points

    # Find arc-length spacing
    alpha = np.linspace(-1,1,num_points)
    D = finite_diff.finite_diff(alpha, 1, fib.num_points_finite_diff)
    arcLength = compute_length(fib.x, D[:,:,1])

    # Upsample the points
    alpha_up = np.linspace(-1,1,num_points_up)
    Pup = bary.barycentricMatrix(alpha, alpha_up)
    xup = np.dot(Pup, fib.x)

    # Find arc-length spacings for the upsampled configuration
    D = finite_diff.finite_diff(alpha_up, 1, fib.num_points_finite_diff)
    arcLengthUp = compute_length(xup, D[:,:,1])

    # Ideal spacing
    h = arcLength[-1] / (num_points-1) 

    # Find spacing * h + remainder = arcLengthUp
    spacings = arcLengthUp // h
    # Find the indices in the upsampled configuration that gives equal spacing
    pointIdcs = []
    allIdcs = np.arange(num_points_up)
    for j in range(num_points):
      if j == 0: 
        pointIdcs.append(0)
      elif j == num_points-1:
        pointIdcs.append(num_points_up-1)
      else:
        arcLengthNext = arcLengthUp[pointIdcs[-1]] + h
        minIdx = np.argsort(np.abs(arcLengthUp-arcLengthNext))
        pointIdcs.append(allIdcs[minIdx[0]])
        #idcs = allIdcs[spacings == j]
        #minIdx = np.argsort(arcLengthUp[idcs])
        #pointIdcs.append(idcs[minIdx[0]])

    # assign equally distributed points in arc-length
    fib.x = xup[pointIdcs,:]

  return fibers
##############################################################################################
def compute_length(x,D_1_0):
  '''
  Compute the length of fiber configuration x
  '''

  x_a = np.dot(D_1_0, x) 
  integrand = np.sqrt(x_a[:,0]**2 + x_a[:,1]**2 + x_a[:,2]**2)

  arc_length = np.zeros_like(integrand)
  alpha = np.linspace(-1,1,integrand.size)

  for k in range(integrand.size):
    if k == integrand.size - 1:
      arc_length[k] = spi.trapz(integrand,alpha)
    else:
      arc_length[k] = spi.trapz(integrand[:k+1],alpha[:k+1])


  return arc_length

##############################################################################################
def find_upsample_rate(fibers, fib_mats, fib_mat_resolutions):
  '''
  Finding how much upsampling is needed to minimize aliasing errorsi
  When there are too many fibers, using smaller max upsampling rate might be 
  better than using numba, because of storage of all related matrices in a list

  For small (a few 100s) number of fibers, using numba is not fast.
  '''

  for k, fib in enumerate(fibers):
    num_points = fib.num_points
    num_points_maxUp = fib.num_points_maxUp
    
    indx = np.where(fib_mat_resolutions == num_points)
    fib_mat = fib_mats[indx[0][0]]

    P_maxUp, P_maxDown, out3, out4 = fib_mat.get_matrices(fib.length, fib.num_points_up, 'P_maxUp')
    D_1_maxUp, D_2_maxUp, D_3_maxUp, D_4_maxUp = fib_mat.get_matrices(fib.length, fib.num_points_up, 'D_maxUps')
    

    # Upsample fiber configuration to the maximum upsampling rate
    x_up = np.dot(P_maxUp, fib.x)

    # Compute the bending term
    xssss_up = np.dot(D_4_maxUp, x_up)

    # Initial upsampling rate
    uprate = 1.5

    # Find the Chebyshev coefficients of xssss_up
    timer.timer('finding_modes_upsample')
    modes = np.empty_like(xssss_up)
    modes[:, 0] = abs(cheb.cheb_calc_coef(xssss_up[:, 0]))/num_points_maxUp
    modes[:, 1] = abs(cheb.cheb_calc_coef(xssss_up[:, 1]))/num_points_maxUp
    modes[:, 2] = abs(cheb.cheb_calc_coef(xssss_up[:, 2]))/num_points_maxUp
    timer.timer('finding_modes_upsample')

    # Compare energy in low and high modes
    cutOff = int(uprate*num_points/2)
    lowEnergy = np.sqrt(np.sum(modes[:cutOff,0]**2 +  modes[:cutOff,1]**2 + \
                               modes[:cutOff,2])**2)
    highEnergy = np.sqrt(np.sum(modes[cutOff:,0]**2 + modes[cutOff:,1]**2 + \
                                modes[cutOff:,2]**2))

    # Increase upsampling rate by 0.5 until energy in lower modes dominates
    desiredRatio = 1.0
    
    while highEnergy/lowEnergy>desiredRatio and uprate< fib.maxUp:
      uprate += 0.5
      cutOff = int(uprate*num_points/2)
      lowEnergy = np.sqrt(np.sum(modes[:cutOff,0]**2 +  modes[:cutOff,1]**2 + \
                                 modes[:cutOff,2])**2)
      highEnergy = np.sqrt(np.sum(modes[cutOff:,0]**2 + modes[cutOff:,1]**2 + \
                                  modes[cutOff:,2]**2))
    
    
    # Update the matrices only when the upsampling rate changes
    if fib.num_points_up != int(num_points * uprate):
      fib.num_points_up = int(num_points * uprate)
      
      # Get upsampled matrices and recompute the followings
      P_up, P_down, out3, out4 = fib_mat.get_matrices(fib.length, fib.num_points_up, 'P_upsample')
      D_1_up, D_2_up, D_3_up, D_4_up = fib_mat.get_matrices(fib.length, fib.num_points_up, 'D_ups')
      P_kerUp, P_kerDn, out3, out4 = fib_mat.get_matrices(fib.length, fib.num_points_up, 'P_kernel')
      
      # Derivatives
      fib.x_up = np.dot(P_up, fib.x)
      fib.xs_up = np.dot(D_1_up, fib.x_up)
      fib.xss_up = np.dot(D_2_up, fib.x_up)
      fib.xsss_up = np.dot(D_3_up, fib.x_up)
      fib.xssss_up = np.dot(D_4_up, fib.x_up)

  return fibers

##############################################################################################

def find_upsample_rate_parallel(fibers, fib_mats, fib_mat_resolutions):
  '''
  Finding how much upsampling is needed to minimize aliasing errors
  Uses numba
  '''
  Nfibers = len(fibers) # number of fibers

  P_maxUps, P_maxDowns, D_4_maxUps = [], [], [] 
  x_fibers, num_points_maxUps, maxUps = [], np.zeros(Nfibers), np.zeros(Nfibers)
  for k, fib in enumerate(fibers):
    indx = np.where(fib_mat_resolutions == fib.num_points)
    fib_mat = fib_mats[indx[0][0]]

    P_maxUp, P_maxDown, out3, out4 = fib_mat.get_matrices(fib.length, fib.num_points_up, 'P_maxUp')
    D_1_maxUp, D_2_maxUp, D_3_maxUp, D_4_maxUp = fib_mat.get_matrices(fib.length, fib.num_points_up, 'D_maxUps')

    P_maxUps.append(P_maxUp)   
    P_maxDowns.append(P_maxDown)   
    D_4_maxUps.append(D_4_maxUp)
    x_fibers.append(fib.x)
    num_points_maxUps[k] = fib.num_points_maxUp
    maxUps[k] = fib.maxUp
  
  num_points_ups = find_upsample_rate_numba_implementation(x_fibers, 
                                                          P_maxUps, 
                                                          P_maxDowns, 
                                                          D_4_maxUps, 
                                                          num_points_maxUps,
                                                          maxUps)

  for k, fib in enumerate(fibers):
    # Update the matrices only when the upsampling rate changes
    if fib.num_points_up != num_points_ups[k]:
      fib.num_points_up = num_points_ups[k]
      indx = np.where(fib_mat_resolutions == fib.num_points)
      fib_mat = fib_mats[indx[0][0]]

      # Get upsampled matrices and recompute the followings
      P_up, P_down, out3, out4 = fib_mat.get_matrices(fib.length, fib.num_points_up, 'P_upsample')
      D_1_up, D_2_up, D_3_up, D_4_up = fib_mat.get_matrices(fib.length, fib.num_points_up, 'D_ups')
      P_kerUp, P_kerDn, out3, out4 = fib_mat.get_matrices(fib.length, fib.num_points_up, 'P_kernel')
      
      # Derivatives
      fib.x_up = np.dot(P_up, fib.x)
      fib.xs_up = np.dot(D_1_up, fib.x_up)
      fib.xss_up = np.dot(D_2_up, fib.x_up)
      fib.xsss_up = np.dot(D_3_up, fib.x_up)
      fib.xssss_up = np.dot(D_4_up, fib.x_up)


  return fibers

##############################################################################################
def find_upsample_rate_numba_implementation(x_fibers, P_maxUps, P_maxDowns, D_4_maxUps, num_points_maxUps, maxUps):

  num_points_ups = np.zeros(num_points_maxUps.size, dtype=np.int32)

  for k in prange(num_points_maxUps.size):
    num_points_maxUp = num_points_maxUps[k]
    xfib = x_fibers[k]
    num_points = xfib.size // 3 
    # Upsample fiber configuration to the maximum upsampling rate
    x_up = np.dot(P_maxUps[k], xfib)

    # Compute the bending term
    xssss_up = np.dot(D_4_maxUps[k], x_up)

    # Initial upsampling rate
    uprate = 1.5

    # Find the Chebyshev coefficients of xssss_up
    modes = np.empty_like(xssss_up)
    
    x, y, z = xssss_up[:,0], xssss_up[:,1], xssss_up[:,2]
    j = np.arange(x.size)
    jk = np.outer(j,j)

    c = np.ones(x.size)
    c[0] = 2.0
    c[-1] = 2.0
    summ = x * np.cos((np.pi / (x.size - 1))*jk) / c
    modes[:,0] = (2.0 / (x.size - 1)) * np.sum(summ, axis=1) / c

    c = np.ones(x.size)
    c[0] = 2.0
    c[-1] = 2.0
    summ = y * np.cos((np.pi / (y.size - 1))*jk) / c
    modes[:,1] = (2.0 / (y.size - 1)) * np.sum(summ, axis=1) / c

    c = np.ones(x.size)
    c[0] = 2.0
    c[-1] = 2.0
    summ =  z * np.cos((np.pi / (z.size - 1))*jk) / c
    modes[:,2] = (2.0 / (z.size - 1)) * np.sum(summ, axis=1) / c

   
    # Compare energy in low and high modes
    cutOff = int(uprate*num_points/2)
    lowEnergy = np.sqrt(np.sum(modes[:cutOff,0]**2 +  modes[:cutOff,1]**2 + \
                               modes[:cutOff,2])**2)
    highEnergy = np.sqrt(np.sum(modes[cutOff:,0]**2 + modes[cutOff:,1]**2 + \
                                modes[cutOff:,2]**2))

    # Increase upsampling rate by 0.5 until energy in lower modes dominates
    desiredRatio = 1.0
    while highEnergy/lowEnergy>desiredRatio and uprate< maxUps[k]:
      uprate += 0.5
      cutOff = int(uprate*num_points/2)
      lowEnergy = np.sqrt(np.sum(modes[:cutOff,0]**2 +  modes[:cutOff,1]**2 + \
                                 modes[:cutOff,2])**2)
      highEnergy = np.sqrt(np.sum(modes[cutOff:,0]**2 + modes[cutOff:,1]**2 + \
                                  modes[cutOff:,2]**2))

    num_points_ups[k] = num_points*uprate

  return num_points_ups

##############################################################################################

def flow_fibers(f, src, trg, fibers, offset_fibers, eta, integration = 'trapz', 
  fib_mats = None, fib_mat_resolutions = None, iupsample = False, 
  oseen_fmm = None, fmm_max_pts = 1000):
  '''
  Flow created by fibers at target points.
  '''
  fw, src_up = [], []
  uprate = fib_mats[0].uprate_poten

  if integration == 'simpsons':
    iupsample = False

  if iupsample:
    src = src.reshape(src.size // 3, 3)
    src_up = np.zeros((offset_fibers[-1]*uprate,3))
    fw = np.zeros((offset_fibers[-1]*uprate,3))
  else:
    fw = np.zeros((offset_fibers[-1],3))

  
  for k, fib in enumerate(fibers):
    # Find the index for fib_mats
    indx = np.where(fib_mat_resolutions == fib.num_points)
    indx = indx[0][0]
    if integration == 'trapz':
      weights, weights_up, out3, out4 = fib_mats[indx].get_matrices(fib.length, fib.num_points_up, 'weights_all')
    else:
      weights, out2, out3, out4 = fib_mats[indx].get_matrices(fib.length, fib.num_points_up, 'simpsons')

    P_kerUp, P_kerDn, out3, out4 = fib_mats[indx].get_matrices(fib.length, fib.num_points_up, 'P_kernel')

    if iupsample:
      weights = weights_up
      fw[uprate*offset_fibers[k]:uprate*offset_fibers[k+1]] = np.dot(P_kerUp,f[offset_fibers[k]:offset_fibers[k+1]])*weights[:,None]
      src_up[uprate*offset_fibers[k]:uprate*offset_fibers[k+1]] = np.dot(P_kerUp,src[offset_fibers[k]:offset_fibers[k+1]])
    else:
      fw[offset_fibers[k]:offset_fibers[k+1]] = f[offset_fibers[k]:offset_fibers[k+1]]*weights[:,None]
  
  fw = fw.flatten()

  if iupsample:
    src = src_up.flatten()

  if oseen_fmm is not None:
    velocity = oseen_fmm(src, trg, fw, eta = eta)
  else:
    velocity = kernels.oseen_kernel_source_target_numba(src, trg, fw, eta = eta)

  return velocity
##############################################################################################

def get_self_fibers_Stokeslet(fibers, eta, fib_mats = None, fib_mat_resolutions = None, iupsample = False):
  '''
  Get the block-diagonal matrix with each block being
  the Oseen kernel for one fiber.
  '''
  G_all = []

  if iupsample:
    for k, fib in enumerate(fibers):
      # Find the index for fib_mats
      indx = np.where(fib_mat_resolutions == fib.num_points)
      indx = indx[0][0]
      P_kerUp, P_kerDn, out3, out4 = fib_mats[indx].get_matrices(fib.length, fib.num_points_up, 'P_kernel')
      # Linear system
      G = kernels.oseen_tensor_source_target(np.dot(P_kerUp,fib.x), fib.x, eta = eta)
      G_all.append(G)
  else:
    for k, fib in enumerate(fibers):
      # Linear system
      G = kernels.oseen_tensor(fib.x, eta = eta)
      G_all.append(G)

  G_block = scsp.block_diag(G_all)
  G = scsp.csr_matrix(G_block)


  return G
##############################################################################################
def no_sliding_cortical_pushing(fibers, offset_fibers, force_fibers, xEnds, nucleus_radius = None, nucleus_position = None, cortex_a = None, cortex_b = None, cortex_c = None, cortex_radius = None):
  attached_ids = np.array([])
  for k, fib in enumerate(fibers):
    fib.iReachSurface = False
  for inuc, radius in enumerate(nucleus_radius):
    d = np.sqrt(xEnds[:,0]**2 + xEnds[:,1]**2 + (xEnds[:,2]-nucleus_position[inuc])**2)
    attached_ids = np.where((d-radius)<=0.15)
    print('Distance to surface: ', d-radius)
  attached_ids = attached_ids[0]
  print('Attached ids: ', attached_ids)
  for idx in attached_ids:
    fibers[idx].iReachSurface = True
    distVec = fibers[idx].x[-1]-np.array([0, 0, nucleus_position[inuc]])
    attached_point = np.array([0, 0, nucleus_position[inuc]]) + distVec/np.linalg.norm(distVec)*radius
    force_fibers[offset_fibers[idx]+fibers[idx].num_points-1] = -10 * (fibers[idx].x[-1]-attached_point)
  return fibers, force_fibers

##############################################################################################

def get_self_fibers_FMMStokeslet(fibers, eta, fib_mats = None, fib_mat_resolutions = None, iupsample = False):
  '''
  Get the block-diagonal matrix with each block being
  the Oseen kernel for one fiber.
  '''
  G_all = []

  if iupsample:
    for k, fib in enumerate(fibers):
      # Find the index for fib_mats
      indx = np.where(fib_mat_resolutions == fib.num_points)
      indx = indx[0][0]
      P_kerUp, P_kerDn, out3, out4 = fib_mats[indx].get_matrices(fib.length, fib.num_points_up, 'P_kernel')
      # Linear system
      G = kernels.oseen_tensor_source_target(np.dot(P_kerUp,fib.x), fib.x, eta = eta)
      np.fill_diagonal(G,0)
      G_all.append(G)
  else:
    for k, fib in enumerate(fibers):
      # Linear system
      G = kernels.oseen_tensor(fib.x, eta = eta)
      np.fill_diagonal(G,0)
      G_all.append(G)

  G_block = scsp.block_diag(G_all)
  G = scsp.csr_matrix(G_block)


  return G
##############################################################################################

def self_flow_fibers(f, offset_fibers, fibers, G, eta, integration = 'trapz', fib_mats = None, fib_mat_resolutions = None, iupsample = False):
  '''
  Flow created by fibers at target points.
  Input: 
  force on fibers (num_points), then upsamples here
  Output:
  v = velocity, dimension (3*num_points)
  '''
  fw = []
  uprate = fib_mats[0].uprate_poten

  if integration == 'simpsons':
    iupsample = False

  if iupsample:
    fw = np.zeros((offset_fibers[-1]*uprate,3))    
    for k, fib in enumerate(fibers):
      # Find the index for fib_mats
      indx = np.where(fib_mat_resolutions == fib.num_points)
      indx = indx[0][0]
      P_kerUp, P_kerDn, out3, out4 = fib_mats[indx].get_matrices(fib.length, fib.num_points_up, 'P_kernel')
      weights, weights_up, out3, out4 = fib_mats[indx].get_matrices(fib.length, fib.num_points_up, 'weights_all')
      # upsample force and multiply with weights
      fw[uprate*offset_fibers[k]:uprate*offset_fibers[k+1]] = np.dot(P_kerUp, f[offset_fibers[k]:offset_fibers[k+1]]) * weights_up[:,None]
  else:
    fw = np.zeros((offset_fibers[-1],3))
    for k, fib in enumerate(fibers):
      # Find the index for fib_mats
      indx = np.where(fib_mat_resolutions == fib.num_points)
      indx = indx[0][0]
      if integration == 'trapz':
        weights, weights_up, out3, out4 = fib_mats[indx].get_matrices(fib.length, fib.num_points_up, 'weights_all')
      else:
        weights, out2, out3, out4 = fib_mats[indx].get_matrices(fib.length, fib.num_points_up, 'simpsons')
      fw[offset_fibers[k]:offset_fibers[k+1]] = f[offset_fibers[k] : offset_fibers[k+1]] * weights[:,None]


  fw = fw.flatten()
  v = -G.dot(fw)

  return v
##############################################################################################
def check_fiber_cortex_interaction(fibers, radius, periphery_a, periphery_b, periphery_c):
  # Cortex could be a sphere or an ellipsoid
  if radius is not None:
    periphery_a, periphery_b, periphery_c = radius, radius, radius
  iReachCortexList = np.zeros(len(fibers),dtype=bool)
  if periphery_a is not None:
    for k, fib in enumerate(fibers):
      xfib, yfib, zfib = fib.x[:,0], fib.x[:,1], fib.x[:,2]
      x = xfib / periphery_a
      y = yfib / periphery_b
      z = zfib / periphery_c

      r_true = np.sqrt(xfib**2 + yfib**2 + zfib**2)

      r_fiber = np.sqrt(x**2 + y**2 + z**2)
      phi_fiber = np.arctan2(y,(x+1e-12))
      theta_fiber = np.arccos(z/(1e-12+r_fiber))

      x_cort = periphery_a*np.sin(theta_fiber)*np.cos(phi_fiber)
      y_cort = periphery_b*np.sin(theta_fiber)*np.sin(phi_fiber)
      z_cort = periphery_c*np.cos(theta_fiber)

      d = np.sqrt((xfib-x_cort)**2 + (yfib-y_cort)**2 + (zfib-z_cort)**2) 
      cortex_point_r = np.sqrt(x_cort**2 + y_cort**2 + z_cort**2)

      sel_out = r_true >= cortex_point_r
      sel_in = d <= 0.1 * cortex_point_r

      if sel_out.any(): iReachCortexList[k] = True
      if sel_in.any(): iReachCortexList[k] = True
  return iReachCortexList
    
##############################################################################################

def flow_bodies(mu, F, tau, src, q, normals, trg, weights, eta):
  '''
  Flow created by bodies at target points.
  q: r_source
  trg: target points including fiber markers
  src: source points (includes what?)
  F: density (force_bodies)
  tau: density in rotlet
  mu: density in stresslet
  normals: normal vector to the
  '''
  v = kernels.oseen_kernel_source_target_numba(q, trg, F, eta = eta)
  v += kernels.rotlet_kernel_source_target_numba(q, trg, tau, eta = eta)
  v += kernels.stresslet_kernel_source_target_numba(src, trg, normals, mu, eta = eta)
  return v

##############################################################################################

def get_num_particles_and_offsets(bodies, fibers, shell, ihydro = False):
  '''
  Calculate numbers of bodies and fibers, offsets 
  '''
  num_bodies = len(bodies)
  num_fibers = len(fibers)
  if ihydro:
    offset_bodies = np.zeros(num_bodies + 1, dtype=int)
    for k, b in enumerate(bodies):
      offset_bodies[k+1] = offset_bodies[k] + b.Nblobs
  else:
    offset_bodies = num_bodies * 6

  offset_fibers = np.zeros(num_fibers + 1, dtype=int)
  for k, fib in enumerate(fibers):
    offset_fibers[k+1] = offset_fibers[k] + fib.num_points

  # Calculate the size of the linear system
  system_size = offset_fibers[-1] * 4 + num_bodies * 6
  if ihydro:
    system_size += offset_bodies[-1] * 3
  if shell is not None:
    system_size += shell.Nblobs * 3

  return num_bodies, num_fibers, offset_bodies, offset_fibers, system_size


##############################################################################################
def build_fibers_force_operator(fibers, fib_mats, fib_mat_resolutions):
  '''
  f = -E * X_ssss + (T*X_s)_s
  '''
  force_operator = []
  for fib in fibers:
    indx = np.where(fib_mat_resolutions == fib.num_points)
    force_operator.append(fib.force_operator(fib_mats[indx[0][0]]))
  force_operator_block = scsp.block_diag(force_operator) 
  
  return scsp.csr_matrix(force_operator_block)


##############################################################################################
def build_link_matrix(system_size,bodies,fibers,offset_fibers,offset_bodies,fib_mats,fib_mat_resolutions):
  '''
  Building link matrix (fibers' boundary conditions)
  '''
  As_dok_BC = scsp.dok_matrix((system_size, system_size))
  for k, b in enumerate(bodies):
    # Get links location
    if b.links_location is not None:
      rotation_matrix = b.orientation.rotation_matrix()
      links_loc = np.array([np.dot(rotation_matrix, vec) for vec in b.links_location])
      offset_links = b.links_first_fibers
    else:
      links_loc = []

    # Loop over attached fibers
    for i, link in enumerate(links_loc):
      offset_fiber = offset_links + i
      offset_point = offset_fibers[offset_fiber] * 4 + offset_bodies
      fib = fibers[offset_fiber]

      # Find the index for fib_mats
      indx = np.where(fib_mat_resolutions == fib.num_points)
      indx = indx[0][0]

      # Get the class that has the matrices
      fib_mat = fib_mats[indx]
      out1, D_2, D_3, out4 = fib_mat.get_matrices(fib.length, fib.num_points_up, 'Ds')
      xs = fib.xs
      

      # Rectangular mathod, Driscoll and Hale
      # Matrix A_body_fiber, for position
      # Force by fiber on body at s = 0, Fext = -F|s=0 = -(EXsss - TXs)
      # Bending term:
      As_dok_BC[k*6 + 0, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] +=  -fib.E * D_3[0,:] 
      As_dok_BC[k*6 + 1, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] +=  -fib.E * D_3[0,:] 
      As_dok_BC[k*6 + 2, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] +=  -fib.E * D_3[0,:] 
      # Tension term:
      As_dok_BC[k*6 + 0, offset_point+fib.num_points*3] += xs[0,0] 
      As_dok_BC[k*6 + 1, offset_point+fib.num_points*3] += xs[0,1] 
      As_dok_BC[k*6 + 2, offset_point+fib.num_points*3] += xs[0,2] 
      
      # Torque by fiber on body at s = 0, Lext = (L + link_loc x F) = -(E(Xss x Xs) + link_loc x (EXsss - TXs))
      # Bending force term:
      # yz:
      As_dok_BC[k*6 + 3, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] += -fib.E * links_loc[i,1] * D_3[0,:] 
      # zy:
      As_dok_BC[k*6 + 3, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] +=  fib.E * links_loc[i,2] * D_3[0,:] 
      # zx:
      As_dok_BC[k*6 + 4, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] += -fib.E * links_loc[i,2] * D_3[0,:] 
      # xz:
      As_dok_BC[k*6 + 4, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] +=  fib.E * links_loc[i,0] * D_3[0,:] 
      # xy:
      As_dok_BC[k*6 + 5, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] += -fib.E * links_loc[i,0] * D_3[0,:]
      # yx:
      As_dok_BC[k*6 + 5, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] +=  fib.E * links_loc[i,1] * D_3[0,:]
      
      # Tension force term:
      # yz - zy:
      As_dok_BC[k*6 + 3, offset_point+fib.num_points*3] += (links_loc[i,1]*xs[0,2] - links_loc[i,2]*xs[0,1])  
      # zx - xz:
      As_dok_BC[k*6 + 4, offset_point+fib.num_points*3] += (links_loc[i,2]*xs[0,0] - links_loc[i,0]*xs[0,2]) 
      # xy - yx:
      As_dok_BC[k*6 + 5, offset_point+fib.num_points*3] += (links_loc[i,0]*xs[0,1] - links_loc[i,1]*xs[0,0]) 

      # Fiber torque (L):
      # yz:
      As_dok_BC[k*6 + 3, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] += -fib.E * xs[0,2] * D_2[0,:] 
      # zy:
      As_dok_BC[k*6 + 3, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] +=  fib.E * xs[0,1] * D_2[0,:] 
      # zx:
      As_dok_BC[k*6 + 4, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] += -fib.E * xs[0,0] * D_2[0,:] 
      # xz:
      As_dok_BC[k*6 + 4, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] +=  fib.E * xs[0,2] * D_2[0,:] 
      # xy:
      As_dok_BC[k*6 + 5, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] += -fib.E * xs[0,1] * D_2[0,:] 
      # yx:
      As_dok_BC[k*6 + 5, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] +=  fib.E * xs[0,0] * D_2[0,:] 
 

      # Matrix A_fiber_body, for position 
      # dx/dt = U + Omega x link_loc (move to LHS so -U -Omega x link_loc)
      # Linear velocity part (U)
      As_dok_BC[offset_point+fib.num_points*4-14, k*6 + 0] += -1.0 
      As_dok_BC[offset_point+fib.num_points*4-13, k*6 + 1] += -1.0 
      As_dok_BC[offset_point+fib.num_points*4-12, k*6 + 2] += -1.0 
      # Angular velocity part (Omega)
      # yz:
      As_dok_BC[offset_point+fib.num_points*4-14, k*6 + 4] += -links_loc[i,2] 
      # zy:
      As_dok_BC[offset_point+fib.num_points*4-14, k*6 + 5] +=  links_loc[i,1] 
      # zx:
      As_dok_BC[offset_point+fib.num_points*4-13, k*6 + 5] += -links_loc[i,0] 
      # xz:
      As_dok_BC[offset_point+fib.num_points*4-13, k*6 + 3] +=  links_loc[i,2] 
      # xy:
      As_dok_BC[offset_point+fib.num_points*4-12, k*6 + 3] += -links_loc[i,1] 
      # yx:
      As_dok_BC[offset_point+fib.num_points*4-12, k*6 + 4] +=  links_loc[i,0] 

      # Tension equation, left hand side of it (U + Omega x link - \bar{u}_f).xs
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 0] += -xs[0,0] 
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 1] += -xs[0,1] 
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 2] += -xs[0,2] 

      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 3] += (xs[0,1]*links_loc[i,2] - xs[0,2]*links_loc[i,1]) 
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 4] += (xs[0,2]*links_loc[i,0] - xs[0,0]*links_loc[i,2]) 
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 5] += (xs[0,0]*links_loc[i,1] - xs[0,1]*links_loc[i,0]) 
      
      
      # Matrix A_fiber_body, for angle 
      # Clamped boundary condition: dXs/dt = Omega x Xs or Omega x link_direction
      link_norm = np.sqrt(links_loc[i,0]**2 + links_loc[i,1]**2 + links_loc[i,2]**2)
      link_dir = links_loc[i,:] / link_norm
      # yz:
      As_dok_BC[offset_point+fib.num_points*4-10, k*6 + 4] += -link_dir[2]
      # zy:
      As_dok_BC[offset_point+fib.num_points*4-10, k*6 + 5] +=  link_dir[1]
      # zx:
      As_dok_BC[offset_point+fib.num_points*4-9, k*6 + 5]  += -link_dir[0]
      # xz:
      As_dok_BC[offset_point+fib.num_points*4-9, k*6 + 3]  +=  link_dir[2]
      # xy:
      As_dok_BC[offset_point+fib.num_points*4-8, k*6 + 3]  += -link_dir[1]
      # yx:
      As_dok_BC[offset_point+fib.num_points*4-8, k*6 + 4]  +=  link_dir[0]
      
      
  return scsp.csr_matrix(As_dok_BC)
##############################################################################################
def build_hard_link_matrix(bodies,fibers,offset_fibers,offset_bodies,fib_mats,fib_mat_resolutions):
  '''
  Building link matrix (fibers' boundary conditions)
  '''
  system_size = 4*offset_fibers[-1] + 6 * len(bodies)
  As_dok_BC = scsp.dok_matrix((system_size, system_size))
  
  # CENTER OF MASS IS PAYLOAD'S POSITION
  center_of_mass = bodies[2].location
  

  for k, b in enumerate(bodies):
    # Get links location
    
    if b.links_location is not None:
      rotation_matrix = b.orientation.rotation_matrix()
      links_loc = np.array([np.dot(rotation_matrix, vec) for vec in b.links_location])
      offset_links = b.links_first_fibers
    else:
      links_loc = []

    # Distance of body's center to CoM
    r2cent = b.location - center_of_mass


    # Loop over attached fibers
    for i, link in enumerate(links_loc):
      offset_fiber = offset_links + i
      offset_point = offset_fibers[offset_fiber] * 4 + offset_bodies
      fib = fibers[offset_fiber]

      # Find the index for fib_mats
      indx = np.where(fib_mat_resolutions == fib.num_points)
      indx = indx[0][0]

      # Get the class that has the matrices
      fib_mat = fib_mats[indx]
      out1, D_2, D_3, out4 = fib_mat.get_matrices(fib.length, fib.num_points_up, 'Ds')
      xs = fib.xs
      
      # Link's location w.r.t center of mass
      link_loc = links_loc[i] + r2cent


      # Rectangular mathod, Driscoll and Hale
      # Matrix A_body_fiber, for position
      # Force by fiber on body at s = 0, Fext = -F|s=0 = -(EXsss - TXs)
      # Bending term:
      As_dok_BC[k*6 + 0, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] +=  -fib.E * D_3[0,:] 
      As_dok_BC[k*6 + 1, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] +=  -fib.E * D_3[0,:] 
      As_dok_BC[k*6 + 2, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] +=  -fib.E * D_3[0,:] 
      # Tension term:
      As_dok_BC[k*6 + 0, offset_point+fib.num_points*3] += xs[0,0] 
      As_dok_BC[k*6 + 1, offset_point+fib.num_points*3] += xs[0,1] 
      As_dok_BC[k*6 + 2, offset_point+fib.num_points*3] += xs[0,2] 
      
      # Torque by fiber on body at s = 0, Lext = (L + link_loc x F) = -(E(Xss x Xs) + link_loc x (EXsss - TXs))
      # Bending force term:
      # yz:
      As_dok_BC[k*6 + 3, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] += -fib.E * link_loc[1] * D_3[0,:] 
      # zy:
      As_dok_BC[k*6 + 3, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] +=  fib.E * link_loc[2] * D_3[0,:] 
      # zx:
      As_dok_BC[k*6 + 4, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] += -fib.E * link_loc[2] * D_3[0,:] 
      # xz:
      As_dok_BC[k*6 + 4, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] +=  fib.E * link_loc[0] * D_3[0,:] 
      # xy:
      As_dok_BC[k*6 + 5, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] += -fib.E * link_loc[0] * D_3[0,:]
      # yx:
      As_dok_BC[k*6 + 5, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] +=  fib.E * link_loc[1] * D_3[0,:]
      
      # Tension force term:
      # yz - zy:
      As_dok_BC[k*6 + 3, offset_point+fib.num_points*3] += (link_loc[1]*xs[0,2] - link_loc[2]*xs[0,1])  
      # zx - xz:
      As_dok_BC[k*6 + 4, offset_point+fib.num_points*3] += (link_loc[2]*xs[0,0] - link_loc[0]*xs[0,2]) 
      # xy - yx:
      As_dok_BC[k*6 + 5, offset_point+fib.num_points*3] += (link_loc[0]*xs[0,1] - link_loc[1]*xs[0,0]) 

      # Fiber torque (L):
      # yz:
      As_dok_BC[k*6 + 3, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] += -fib.E * xs[0,2] * D_2[0,:] 
      # zy:
      As_dok_BC[k*6 + 3, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] +=  fib.E * xs[0,1] * D_2[0,:] 
      # zx:
      As_dok_BC[k*6 + 4, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] += -fib.E * xs[0,0] * D_2[0,:] 
      # xz:
      As_dok_BC[k*6 + 4, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] +=  fib.E * xs[0,2] * D_2[0,:] 
      # xy:
      As_dok_BC[k*6 + 5, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] += -fib.E * xs[0,1] * D_2[0,:] 
      # yx:
      As_dok_BC[k*6 + 5, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] +=  fib.E * xs[0,0] * D_2[0,:] 
 

      # Matrix A_fiber_body, for position 
      # dx/dt = U + Omega x link_loc (move to LHS so -U -Omega x link_loc)
      # Linear velocity part (U)
      As_dok_BC[offset_point+fib.num_points*4-14, k*6 + 0] += -1.0 
      As_dok_BC[offset_point+fib.num_points*4-13, k*6 + 1] += -1.0 
      As_dok_BC[offset_point+fib.num_points*4-12, k*6 + 2] += -1.0 
      # Angular velocity part (Omega)
      # yz:
      As_dok_BC[offset_point+fib.num_points*4-14, k*6 + 4] += -link_loc[2] 
      # zy:
      As_dok_BC[offset_point+fib.num_points*4-14, k*6 + 5] +=  link_loc[1] 
      # zx:
      As_dok_BC[offset_point+fib.num_points*4-13, k*6 + 5] += -link_loc[0] 
      # xz:
      As_dok_BC[offset_point+fib.num_points*4-13, k*6 + 3] +=  link_loc[2] 
      # xy:
      As_dok_BC[offset_point+fib.num_points*4-12, k*6 + 3] += -link_loc[1] 
      # yx:
      As_dok_BC[offset_point+fib.num_points*4-12, k*6 + 4] +=  link_loc[0] 

      # Tension equation, left hand side of it (U + Omega x link - \bar{u}_f).xs
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 0] += -xs[0,0] 
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 1] += -xs[0,1] 
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 2] += -xs[0,2] 

      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 3] += (xs[0,1]*link_loc[2] - xs[0,2]*link_loc[1]) 
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 4] += (xs[0,2]*link_loc[0] - xs[0,0]*link_loc[2]) 
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 5] += (xs[0,0]*link_loc[1] - xs[0,1]*link_loc[0]) 
      
      
      # Matrix A_fiber_body, for angle 
      # Clamped boundary condition: dXs/dt = Omega x Xs or Omega x link_direction
      link_norm = np.sqrt(link_loc[0]**2 + link_loc[1]**2 + link_loc[2]**2)
      link_dir = link_loc / link_norm
      xs0 = xs[0,:]
      # yz:
      As_dok_BC[offset_point+fib.num_points*4-10, k*6 + 4] += -xs0[2]
      # zy:
      As_dok_BC[offset_point+fib.num_points*4-10, k*6 + 5] +=  xs0[1]
      # zx:
      As_dok_BC[offset_point+fib.num_points*4-9, k*6 + 5]  += -xs0[0]
      # xz:
      As_dok_BC[offset_point+fib.num_points*4-9, k*6 + 3]  +=  xs0[2]
      # xy:
      As_dok_BC[offset_point+fib.num_points*4-8, k*6 + 3]  += -xs0[1]
      # yx:
      As_dok_BC[offset_point+fib.num_points*4-8, k*6 + 4]  +=  xs0[0]
      
      
  return scsp.csr_matrix(As_dok_BC)
##############################################################################################
def build_link_matrix_old(system_size,bodies,fibers,offset_fibers,offset_bodies,fib_mats,fib_mat_resolutions):
  '''
  Building link matrix (fibers' boundary conditions)
  '''
  As_dok_BC = scsp.dok_matrix((system_size, system_size))
  for k, b in enumerate(bodies):
    # Get links location
    if b.links_location is not None:
      rotation_matrix = b.orientation.rotation_matrix()
      links_loc = np.array([np.dot(rotation_matrix, vec) for vec in b.links_location])
      offset_links = b.links_first_fibers
    else:
      links_loc = []

    # Loop over attached fibers
    for i, link in enumerate(links_loc):
      offset_fiber = offset_links + i
      offset_point = offset_fibers[offset_fiber] * 4 + offset_bodies
      fib = fibers[offset_fiber]

      # Find the index for fib_mats
      indx = np.where(fib_mat_resolutions == fib.num_points)
      indx = indx[0][0]

      # Get the class that has the matrices
      fib_mat = fib_mats[indx]
      out1, D_2, D_3, out4 = fib_mat.get_matrices(fib.length, fib.num_points_up, 'Ds')
      xs = fib.xs
      

      # Rectangular mathod, Driscoll and Hale
      # Matrix A_body_fiber, for position
      As_dok_BC[k*6 + 0, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] = fib.E * D_3[0,:] 
      As_dok_BC[k*6 + 1, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] = fib.E * D_3[0,:] 
      As_dok_BC[k*6 + 2, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] = fib.E * D_3[0,:] 
      As_dok_BC[k*6 + 0, offset_point+fib.num_points*3] = -xs[0,0] 
      As_dok_BC[k*6 + 1, offset_point+fib.num_points*3] = -xs[0,1] 
      As_dok_BC[k*6 + 2, offset_point+fib.num_points*3] = -xs[0,2] 
      
      # Original
      As_dok_BC[k*6 + 3, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] = -fib.E * links_loc[i,2] * D_3[0,:] 
      As_dok_BC[k*6 + 3, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] =  fib.E * links_loc[i,1] * D_3[0,:] 
      As_dok_BC[k*6 + 4, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] =  fib.E * links_loc[i,2] * D_3[0,:] 
      As_dok_BC[k*6 + 4, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] = -fib.E * links_loc[i,0] * D_3[0,:] 
      As_dok_BC[k*6 + 5, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] = -fib.E * links_loc[i,1] * D_3[0,:] 
      As_dok_BC[k*6 + 5, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] =  fib.E * links_loc[i,0] * D_3[0,:]
      
      As_dok_BC[k*6 + 3, offset_point+fib.num_points*3] = (-links_loc[i,2]*xs[0,1]+links_loc[i,1]*xs[0,2]) 
      As_dok_BC[k*6 + 4, offset_point+fib.num_points*3] = ( links_loc[i,2]*xs[0,0]-links_loc[i,0]*xs[0,2]) 
      As_dok_BC[k*6 + 5, offset_point+fib.num_points*3] = (-links_loc[i,1]*xs[0,0]+links_loc[i,0]*xs[0,1]) 

      # Matrix A_fiber_body, for position 
      As_dok_BC[offset_point+fib.num_points*4-14, k*6 + 0] = -1.0 
      As_dok_BC[offset_point+fib.num_points*4-13, k*6 + 1] = -1.0 
      As_dok_BC[offset_point+fib.num_points*4-12, k*6 + 2] = -1.0 
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 0] = -xs[0,0] 
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 1] = -xs[0,1] 
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 2] = -xs[0,2] 
      
      As_dok_BC[offset_point+fib.num_points*4-14, k*6 + 4] = -links_loc[i,2] 
      As_dok_BC[offset_point+fib.num_points*4-14, k*6 + 5] =  links_loc[i,1] 
      As_dok_BC[offset_point+fib.num_points*4-13, k*6 + 3] =  links_loc[i,2] 
      As_dok_BC[offset_point+fib.num_points*4-13, k*6 + 5] = -links_loc[i,0] 
      As_dok_BC[offset_point+fib.num_points*4-12, k*6 + 3] = -links_loc[i,1] 
      As_dok_BC[offset_point+fib.num_points*4-12, k*6 + 4] =  links_loc[i,0] 
      
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 3] = ( xs[0,1]*links_loc[i,2] - xs[0,2]*links_loc[i,1]) 
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 4] = (-xs[0,0]*links_loc[i,2] + xs[0,2]*links_loc[i,0]) 
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 5] = ( xs[0,0]*links_loc[i,1] - xs[0,1]*links_loc[i,0]) 

      
      # Matrix A_body_fiber, for angle 
      # Original
      As_dok_BC[k*6 + 3, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] +=  fib.E * xs[0,2] * D_2[0,:] 
      As_dok_BC[k*6 + 3, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] += -fib.E * xs[0,1] * D_2[0,:] 
      As_dok_BC[k*6 + 4, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] += -fib.E * xs[0,2] * D_2[0,:] 
      As_dok_BC[k*6 + 4, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] +=  fib.E * xs[0,0] * D_2[0,:] 
      As_dok_BC[k*6 + 5, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] +=  fib.E * xs[0,1] * D_2[0,:] 
      As_dok_BC[k*6 + 5, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] += -fib.E * xs[0,0] * D_2[0,:] 
      
      # Matrix A_fiber_body, for angle 
      link_norm = np.sqrt(links_loc[i,0]**2 + links_loc[i,1]**2 + links_loc[i,2]**2)
      link_dir = links_loc[i,:] / link_norm
     
      As_dok_BC[offset_point+fib.num_points*4-10, k*6 + 4] = -link_dir[2]
      As_dok_BC[offset_point+fib.num_points*4-10, k*6 + 5] =  link_dir[1]
      As_dok_BC[offset_point+fib.num_points*4-9, k*6 + 3]  =  link_dir[2]
      As_dok_BC[offset_point+fib.num_points*4-9, k*6 + 5]  = -link_dir[0]
      As_dok_BC[offset_point+fib.num_points*4-8, k*6 + 3]  = -link_dir[1]
      As_dok_BC[offset_point+fib.num_points*4-8, k*6 + 4]  =  link_dir[0]

  return scsp.csr_matrix(As_dok_BC)

  ##############################################################################################

def get_fibers_and_bodies_matrices(fibers, bodies, shell, system_size, 
  offset_fibers, offset_bodies, force_fibers, motor_force_fibers, force_bodies, v_on_fibers, v_on_bodies, v_on_shell, 
  fib_mats, fib_mat_resolutions, inextensibility = 'penalty', BC_start_0='force', BC_start_vec_0 = np.zeros(3), BC_start_1='torque', 
  BC_end_0='force', BC_end_vec_0 = np.zeros(3), ihydro = False):
  
  '''
  Prepares RHS for a system of fibers, bodies, periphery with or without hydro
  ihydro is flag for whether there is flow or not
  '''

  A_all = [np.zeros((1,1)) for n in range(len(fibers))]
  RHS_all = np.zeros(system_size)

  if v_on_fibers is not None:
    v_on_fibers = v_on_fibers.reshape(v_on_fibers.size//3, 3) 

  if shell is not None:
    RHS_all[:3*shell.Nblobs] = -v_on_shell
    offset = 3*shell.Nblobs
  else:
    offset = 0

  if ihydro:
    for k, b in enumerate(bodies):
      istart = offset + 3*offset_bodies[k]+6*k
      RHS_all[istart:istart+3*b.Nblobs] = -v_on_bodies[3*offset_bodies[k]:3*offset_bodies[k+1]] 
  else:
    if force_bodies.any():
      RHS_all[0:6*len(bodies):6] = force_bodies[:,0]
      RHS_all[1:6*len(bodies):6] = force_bodies[:,1]
      RHS_all[2:6*len(bodies):6] = force_bodies[:,2]

  for k, fib in enumerate(fibers):

    # Find the index for fib_mats
    indx = np.where(fib_mat_resolutions == fib.num_points)
    indx = indx[0][0]

    # Get the class that has the matrices
    fib_mat = fib_mats[indx]
    A = fib.form_linear_operator(fib_mat, inextensibility = inextensibility) 
    
    
    flow_on = None
    if v_on_fibers is not None:
      flow_on = v_on_fibers[offset_fibers[k]:offset_fibers[k+1]] 
    
    force_on = None
    if force_fibers.any():
      force_on = force_fibers[offset_fibers[k]:offset_fibers[k+1]]

    motor_force_on = None
    if motor_force_fibers.any():
      motor_force_on = motor_force_fibers[offset_fibers[k]:offset_fibers[k+1]]
      
    tot_force_on = None
    if motor_force_on is not None:
      tot_force_on = motor_force_on
      if force_on is not None:
        tot_force_on += force_on
    else:
      if force_on is not None:
        tot_force_on = force_on
    # DONE
    RHS = fib.compute_RHS(fib_mat = fib_mat, force_external = tot_force_on, flow = flow_on, inextensibility = inextensibility)

    # Set BCs if they are given
    if fib.iReachSurface: # Hinged
      print(k, 'th fiber is hinged at the surface')
      BC_end_1 = 'torque'
      BC_end_vec_1 = np.zeros(3)
      BC_end_0 = 'velocity'
      BC_end_vec_0 = np.zeros(3)
    else:
      BC_end_1 = 'torque'
      BC_end_vec_1 = np.zeros(3)
      BC_end_0 = 'force'
      BC_end_vec_0 = np.zeros(3)
      if force_on is not None:
        BC_end_vec_0 = force_on[-1]

    fib.set_BC(BC_start_0=BC_start_0,
      BC_start_1=BC_start_1,
      BC_end_0=BC_end_0,
      BC_end_vec_0=BC_end_vec_0,
      BC_end_1 = BC_end_1,
      BC_end_vec_1 = BC_end_vec_1)

    # Apply BC
    A, RHS = fib.apply_BC_rectangular(A, RHS, fib_mat, flow_on, tot_force_on) # DONE

    # Save data
    A_all[k] = A
    if ihydro:
      istart = offset + offset_bodies[-1]*3 + 6*len(bodies) + 4*offset_fibers[k]
    else:
      istart = 6*len(bodies) + 4*offset_fibers[k]

    RHS_all[istart : istart + 4*fib.num_points] = RHS

  # Transform to sparse matrix
  if len(A_all) > 0:
    As_fibers_block = scsp.block_diag(A_all)
    As_fibers = scsp.csr_matrix(As_fibers_block)
  else:
    As_fibers, As_fibers_block = None, None

  return As_fibers, A_all, RHS_all



##############################################################################################
def build_block_diagonal_preconditioner_body(bodies, eta, K_bodies = None):
  '''
  Block diagonal preconditioner for rigid bodies 
  '''

  LU_all = []
  P_all = []
  A_inv_all = []
  for k, b in enumerate(bodies):
    
    r_vectors = b.get_r_vectors_surface()
    normals = b.get_normals()
    weights = b.quadrature_weights
    M = kernels.stresslet_kernel_times_normal_numba(r_vectors, normals, eta)
    if True:
      # Using singularity subtraction D[q - q(x)](x)
      ex, ey, ez = b.ex.flatten(), b.ey.flatten(), b.ez.flatten()

      I = np.zeros((3*b.Nblobs, 3*b.Nblobs))
      for i in range(b.Nblobs):
        I[3*i:3*(i+1), 3*i+0] = ex[3*i:3*(i+1)] / weights[i]
        I[3*i:3*(i+1), 3*i+1] = ey[3*i:3*(i+1)] / weights[i]
        I[3*i:3*(i+1), 3*i+2] = ez[3*i:3*(i+1)] / weights[i]
      M -= I
    else:
      # Using naive representation -0.5*q(x) + D[q](x)
      I_vec = np.ones(b.Nblobs*3)
      I_vec[0::3] /= (2.0 * weights)
      I_vec[1::3] /= (2.0 * weights)
      I_vec[2::3] /= (2.0 * weights)
      M -= np.diag(I_vec)

    A = np.zeros((3*b.Nblobs+6, 3*b.Nblobs+6))
    if K_bodies is None:
      K = b.calc_K_matrix()
    else:
      K = K_bodies[k]

    A[0:3*b.Nblobs, 0:3*b.Nblobs] = M
    A[0:3*b.Nblobs, 3*b.Nblobs:3*b.Nblobs+6] = -K
    A[3*b.Nblobs:3*b.Nblobs+6, 0:3*b.Nblobs] = -K.T
    A[3*b.Nblobs:3*b.Nblobs+6, 3*b.Nblobs:3*b.Nblobs+6] = np.eye(6)
    (LU, P) = scla.lu_factor(A)
    P_all.append(P)
    LU_all.append(LU)

  return LU_all, P_all
##############################################################################################
def build_block_diagonal_preconditioner_fiber(fibers, A_fibers_blocks):

  '''
  Block diagonal preconditioner for fibers based on Q,R factorization
  '''
  
  Q_all = [np.zeros((1,1)) for n in range(len(A_fibers_blocks))] 
  R_all = [np.zeros((1,1)) for n in range(len(A_fibers_blocks))]

  for k in range(len(A_fibers_blocks)):
    Q, R = scla.qr(A_fibers_blocks[k], check_finite=False)
    Q_all[k] = Q
    R_all[k] = R
    
  return Q_all, R_all
##############################################################################################
def build_block_diagonal_lu_preconditioner_fiber(fibers, A_fibers_blocks):

  '''
  Block diagonal preconditioner for fibers based on Q,R factorization
  '''
  
  LU_all = [np.zeros((1,1)) for n in range(len(A_fibers_blocks))] 
  P_all = [np.zeros((1,1)) for n in range(len(A_fibers_blocks))]

  for k in range(len(A_fibers_blocks)):
    LU, P = scla.lu_factor(A_fibers_blocks[k], check_finite=False)
    LU_all[k] = LU
    P_all[k] = P
    
  return LU_all, P_all
##############################################################################################

def K_matrix_vector_prod(bodies, vector, offset_bodies, K_bodies = None, Transpose = False):
  '''
  Compute the matrix vector product K*vector where
  K is the geometrix matrix that transport the information from the 
  level of describtion of the body to the level of describtion of the blobs.
  ''' 
  # Prepare variables
  if Transpose:
    result = np.empty((len(bodies), 6))
    v = np.reshape(vector, (offset_bodies[-1] * 3))
  else:
    result = np.empty((offset_bodies[-1], 3))
    v = np.reshape(vector, (len(bodies)*6))

  # Loop over bodies
  offset = 0
  for k, b in enumerate(bodies):
    if K_bodies is None:
      K = b.calc_K_matrix()
    else:
      K = K_bodies[k]
    if Transpose:
      result[k] = np.dot(K.T, v[3*offset : 3*(offset+b.Nblobs)])
    else:
      result[offset : offset+b.Nblobs] = np.reshape(np.dot(K, v[6*k : 6*(k+1)]), (b.Nblobs, 3))
    offset += b.Nblobs    
  return result

##############################################################################################

def get_blobs_normals(bodies, offset_bodies):
  '''
  Return coordinates of all the blobs with shape (Nblobs, 3).
  '''
  for b in bodies:
    normals[offset:(offset+b.Nblobs)] = b.get_normals()
    offset += b.Nblobs
  return normals

##############################################################################################
