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
def dynamic_instability(fibers,
                        bodies,
                        prams,
                        options,
                        dt,
                        radius_a,
                        radius_b,
                        radius_c,
                        current_time,
                        molecular_motors = None):
  '''
  Dynamic instability of MTs
  '''
  nucleation_rate = prams.nucleation_rate
  rate_catastrophe = prams.rate_catastrophe
  v_growth = prams.v_growth
  fiber_ds = options.fiber_ds
  minL = 0.5

  fibers_gone = []
  sites_gone_imaginary = []
  terminal_time_gone_imaginary = []
  # Check if fiber reaches cortex, if so, then it goes through catastrophe
  # Otherwise polymerize, depolymerize

  temp_time, temp_idcs = [], []
  for k, b in enumerate(bodies):
    num_sites = len(b.occupied_site_idcs)
    for idx in range(num_sites):
      if current_time >= b.when_empty_occupied_sites[idx]:
        sites_gone_imaginary.append(b.occupied_site_idcs[idx])
        terminal_time_gone_imaginary.append(b.when_empty_occupied_sites[idx])
        b.passive_sites_idcs.append(b.occupied_site_idcs[idx])
      else:
        temp_time.append(b.when_empty_occupied_sites[idx])
        temp_idcs.append(b.occupied_site_idcs[idx])
  b.when_empty_occupied_sites = temp_time
  b.occupied_site_idcs = temp_idcs


  temp_fibers = []
  for k, fib in enumerate(fibers):
    xfib, yfib, zfib = fib.x[:,0], fib.x[:,1], fib.x[:,2]
    iReachSurface = False

    # If there is a cortex, then check if MT reaches it
    # No catastrophe upon reaching cortex in Cort-Push NoSliding


    if not iReachSurface:
      # Poisson process for CATASTROPHE
      max_length = prams.max_length
      if max_length is not None:
        if fib.length >= max_length and fib.length_before_die is not None:
          max_length = fib.length_before_die * prams.scale_life_time
        elif fib.length < max_length and fib.length_before_die is not None:
          max_length = fib.length_before_die
      else:
        max_length = fib.length_before_die
        if fib.iReachSurface: max_length *= prams.scale_life_time

      if max_length is not None and fib.length >= max_length:
        # Go through CATASTROPHE
        idx_body = fib.attached_to_body
        fibers_gone.append(fibers[k])
        bodies[idx_body].passive_sites_idcs.append(fib.nuc_site_idx)
        nuc_site_idx_in_active = bodies[idx_body].active_sites_idcs.index(fib.nuc_site_idx)
        del bodies[idx_body].active_sites_idcs[nuc_site_idx_in_active]
      else:
        # otherwise grow with v_growth
        v_length = v_growth
        if fib.iReachSurface: v_length = prams.scale_vg * v_growth
        if prams.max_length is not None and fib.length >= prams.max_length: v_length = 0
        fib.v_length = v_length
        fib.length_previous = np.copy(fib.length)
        fib.length = fib.length + fib.v_length * dt
        temp_fibers.append(fib)
  # Remove fibers
  fibers = temp_fibers
  # Poisson process to nucleate new MTs
  Ninit = options.num_points

  s = np.linspace(0, 2, Ninit)

  num_nucleated = 0
  num_nucleated_imaginary = 0
  for ib, b in enumerate(bodies):
    # Get body's position and orientation
    center, orientation = b.location, b.orientation
    rotation_matrix = orientation.rotation_matrix()

    # Put locations of occupied sites into an array
    # Reference config.
    active_sites = np.array([])
    for k in np.arange(len(b.active_sites_idcs)):
      loc = b.nuc_sites[b.active_sites_idcs[k]]
      if active_sites.any():
        active_sites = np.concatenate((active_sites,np.reshape(loc,(1,3))),axis=0)
      else:
        active_sites = np.reshape(loc,(1,3))
    # Find number of MTs to nucleate
    if prams.max_nuc_sites is not None:
      num_empty_sites = min(prams.max_nuc_sites-len(b.active_sites_idcs), len(b.passive_sites_idcs))
      num_to_nucleate = min(np.random.poisson(nucleation_rate * dt), num_empty_sites)
    else:
      num_to_nucleate = min(np.random.poisson(nucleation_rate * dt), len(b.passive_sites_idcs))
    # Nucleate MTs
    for k in np.arange(num_to_nucleate):
      cannot_place_fiber = True
      ntrial = 0
      while cannot_place_fiber and ntrial < 50:
        # choose a link
        idx_in_passive = np.random.randint(len(b.passive_sites_idcs))
        idx_in_all = b.passive_sites_idcs[idx_in_passive]
        ilink = b.nuc_sites[idx_in_all]
        site_location = np.dot(rotation_matrix, ilink)
        site_normal = ilink / np.linalg.norm(ilink)

        # if there active links, then check interfilament spacing
        if active_sites.size > 0:
          dummy_links = np.concatenate((active_sites, np.reshape(ilink, (1,3))), axis=0)
          dx = dummy_links[:,0] - dummy_links[:,0,None]
          dy = dummy_links[:,1] - dummy_links[:,1,None]
          dz = dummy_links[:,2] - dummy_links[:,2,None]
          dr = np.sqrt(dx**2 + dy**2 + dz**2)
          dfilament = min(dr[0,1:])
          if dfilament > b.min_ds:
            cannot_place_fiber = False
          else:
            ntrial += 1
        else:
          # if there is no active site, then nucleate anywhere
          cannot_place_fiber = False

      # FIX ME: This would nucleate only around the belly
      if not cannot_place_fiber:
        if b.radius == 0.5: zone_h = 0.2
        if b.radius == 5: zone_h = 0.5
        if abs(ilink[2]) > zone_h:
          b.occupied_site_idcs.append(idx_in_all)
          terminal_time = current_time + np.random.exponential(1/rate_catastrophe)
          b.when_empty_occupied_sites.append(terminal_time)
          del b.passive_sites_idcs[idx_in_passive]
          num_nucleated_imaginary += 1
          cannot_place_fiber = True

      if not cannot_place_fiber:
        # save occupied link location
        if active_sites.size > 0:
          active_sites = np.concatenate((active_sites,np.reshape(ilink,(1,3))), axis=0)
        else:
          active_sites = np.reshape(ilink,(1,3))
        # update active and remaining link lists:
        del b.passive_sites_idcs[idx_in_passive]
        b.active_sites_idcs.append(idx_in_all)

        # nucleate MT with initial length if minL
        axis = np.dot(rotation_matrix, site_normal)
        axis_s = np.empty((s.size, 3))
        axis_s[:, 0] = axis[0] * s
        axis_s[:, 1] = axis[1] * s
        axis_s[:, 2] = axis[2] * s
        axis_s = axis_s * (minL / 2.0) + site_location + center

        # add new fiber to the fibers list
        fib = fiber.fiber(
            num_points=Ninit,
            fiber_ds = fiber_ds,
            num_points_max=options.num_points_max,
            num_points_finite_diff=options.num_points_finite_diff,
            dt=dt,
            E=prams.Efib,
            length=minL,
            adaptive_num_points=options.adaptive_num_points,
            growing=prams.growing,
            viscosity=prams.viscosity,
            BC_start_0='velocity',
            BC_start_1='angular_velocity')
        fib.force_stall = prams.force_stall
        fib.attached_to_body = ib
        fib.nuc_site_idx = idx_in_all
        fib.ID = b.ID
        fib.x = axis_s
        fib.v_length = v_growth
        fib.length += fib.v_length * dt
        # randomly assing life time from exponential distribution
        if rate_catastrophe > 0:
          fib.length_before_die = v_growth * np.random.exponential(1/rate_catastrophe)
        fibers.append(fib)
        num_nucleated += 1

  return fibers, fibers_gone, bodies, num_nucleated, num_nucleated_imaginary, sites_gone_imaginary, terminal_time_gone_imaginary

##############################################################################################
def dynamic_instability_v2(fibers,
                        bodies,
                        prams,
                        options,
                        dt,
                        radius_a,
                        radius_b,
                        radius_c,
                        current_time,
                        molecular_motors = None):
  '''
  Dynamic instability of MTs
  '''
  nucleation_rate = prams.nucleation_rate
  rate_catastrophe = prams.rate_catastrophe
  v_growth = prams.v_growth
  fiber_ds = options.fiber_ds
  minL = prams.minL

  fibers_gone = []
  sites_gone_imaginary = []
  terminal_time_gone_imaginary = []
  # Check if fiber reaches cortex, if so, then it goes through catastrophe
  # Otherwise polymerize, depolymerize

  # FIBERS IN THE IMAGINARY SITES ARE FREE TO GROW, AWAY FROM SURFACE

  for k, b in enumerate(bodies):
    temp_idcs = []
    num_sites = len(b.occupied_site_idcs)
    for idx in range(num_sites):
      r = np.random.rand(1)
      if r > np.exp(-dt * rate_catastrophe):
        sites_gone_imaginary.append(b.occupied_site_idcs[idx])
        b.passive_sites_idcs.append(b.occupied_site_idcs[idx])
      else:
        temp_idcs.append(b.occupied_site_idcs[idx])
    b.occupied_site_idcs = temp_idcs


  temp_fibers = []
  site_idcs_dying = []
  for k, fib in enumerate(fibers):
    # Sample a random number
    r = np.random.rand(1)

    v_length = v_growth
    r_cat_fib = rate_catastrophe
    if fib.iReachSurface or fib.iReachSurface_fake:
      v_length = v_growth * prams.scale_vg
      r_cat_fib = rate_catastrophe / prams.scale_life_time

    if r > np.exp(-dt * r_cat_fib): # then fiber goes through CATASTROPHE
      site_idcs_dying.append(fib.nuc_site_idx)
      idx_body = fib.attached_to_body
      fibers_gone.append(fibers[k])
      bodies[idx_body].passive_sites_idcs.append(fib.nuc_site_idx)
      nuc_site_idx_in_active = bodies[idx_body].active_sites_idcs.index(fib.nuc_site_idx)
      del bodies[idx_body].active_sites_idcs[nuc_site_idx_in_active]
    else:
      # otherwise grow with v_growth
      if prams.max_length is not None and fib.length >= prams.max_length: v_length = 0

      # Check if the fiber nucleated at |z|>0.25 reaches the cortex

      ilink = bodies[fib.attached_to_body].nuc_sites[fib.nuc_site_idx]
      center, orientation = b.location, b.orientation
      rotation_matrix = orientation.rotation_matrix()
      nuc_site_xyz = np.dot(rotation_matrix, ilink)
      fib.stop_growing = False
      body_radius = bodies[fib.attached_to_body].radius
      if abs(nuc_site_xyz[2]) > body_radius/2:
        xfib, yfib, zfib = fib.x[:,0], fib.x[:,1], fib.x[:,2]
        x = xfib[-1]/radius_a
        y = yfib[-1]/radius_b
        z = zfib[-1]/radius_c

        r_true = np.sqrt(xfib[-1]**2 + yfib[-1]**2 + zfib[-1]**2)
        r_fiber = np.sqrt(x**2 + y**2 + z**2)
        phi_fiber = np.arctan2(y, (x + 1e-12))
        theta_fiber = np.arccos(z / (1e-12 + r_fiber))

        x_cort = radius_a * np.sin(theta_fiber) * np.cos(phi_fiber)
        y_cort = radius_b * np.sin(theta_fiber) * np.sin(phi_fiber)
        z_cort = radius_c * np.cos(theta_fiber)

        d2cort = np.sqrt((xfib[-1] - x_cort)**2 + (yfib[-1] - y_cort)**2 + (zfib[-1] - z_cort)**2)

        if d2cort <= 1.5: fib.stop_growing = True

      if fib.stop_growing: v_length = 0
      fib.v_length = v_length
      fib.length_previous = np.copy(fib.length)
      fib.length = fib.length + fib.v_length * dt
      temp_fibers.append(fib)
  # Remove fibers
  fibers = temp_fibers
  # Poisson process to nucleate new MTs
  Ninit = options.num_points

  s = np.linspace(0, 2, Ninit)

  num_nucleated = 0
  site_idcs_nucleating = []
  num_nucleated_imaginary = 0
  for ib, b in enumerate(bodies):
    # Get body's position and orientation
    center, orientation = b.location, b.orientation
    rotation_matrix = orientation.rotation_matrix()

    # Put locations of occupied sites into an array
    # Reference config.
    active_sites = np.array([])
    for k in np.arange(len(b.active_sites_idcs)):
      loc = b.nuc_sites[b.active_sites_idcs[k]]
      if active_sites.any():
        active_sites = np.concatenate((active_sites,np.reshape(loc,(1,3))),axis=0)
      else:
        active_sites = np.reshape(loc,(1,3))
    # Find number of MTs to nucleate
    num_to_nucleate = min(np.random.poisson(nucleation_rate * dt), len(b.passive_sites_idcs))
    # Nucleate MTs
    for k in np.arange(num_to_nucleate):
      cannot_place_fiber = True
      ntrial = 0
      while cannot_place_fiber and ntrial < 50:
        # choose a link
        idx_in_passive = np.random.randint(len(b.passive_sites_idcs))
        idx_in_all = b.passive_sites_idcs[idx_in_passive]
        ilink = b.nuc_sites[idx_in_all]
        site_location = np.dot(rotation_matrix, ilink)
        site_normal = ilink / np.linalg.norm(ilink)

        # if there active links, then check interfilament spacing
        if active_sites.size > 0:
          dummy_links = np.concatenate((active_sites, np.reshape(ilink, (1,3))), axis=0)
          dx = dummy_links[:,0] - dummy_links[:,0,None]
          dy = dummy_links[:,1] - dummy_links[:,1,None]
          dz = dummy_links[:,2] - dummy_links[:,2,None]
          dr = np.sqrt(dx**2 + dy**2 + dz**2)
          dfilament = min(dr[0,1:])
          if dfilament > b.min_ds:
            cannot_place_fiber = False
          else:
            ntrial += 1
        else:
          # if there is no active site, then nucleate anywhere
          cannot_place_fiber = False

      # FIX ME: This would nucleate only around the belly
      if not cannot_place_fiber:
        if b.radius == 0.5: zone_h = 0.2
        if b.radius == 5: zone_h = 0.5
        if False: #abs(site_location[2]) > zone_h:
          b.occupied_site_idcs.append(idx_in_all)
          del b.passive_sites_idcs[idx_in_passive]
          num_nucleated_imaginary += 1
          cannot_place_fiber = True

      if not cannot_place_fiber:
        # save occupied link location
        if active_sites.size > 0:
          active_sites = np.concatenate((active_sites,np.reshape(ilink,(1,3))), axis=0)
        else:
          active_sites = np.reshape(ilink,(1,3))
        # update active and remaining link lists:
        del b.passive_sites_idcs[idx_in_passive]
        b.active_sites_idcs.append(idx_in_all)

        # nucleate MT with initial length if minL
        axis = np.dot(rotation_matrix, site_normal)
        axis_s = np.empty((s.size, 3))
        axis_s[:, 0] = axis[0] * s
        axis_s[:, 1] = axis[1] * s
        axis_s[:, 2] = axis[2] * s
        axis_s = axis_s * (minL / 2.0) + site_location + center

        # add new fiber to the fibers list
        fib = fiber.fiber(
            num_points=Ninit,
            fiber_ds = fiber_ds,
            num_points_max=options.num_points_max,
            num_points_finite_diff=options.num_points_finite_diff,
            dt=dt,
            E=prams.Efib,
            length=minL,
            adaptive_num_points=options.adaptive_num_points,
            growing=prams.growing,
            viscosity=prams.viscosity,
            BC_start_0='velocity',
            BC_start_1='angular_velocity')
        fib.force_stall = prams.force_stall
        fib.attached_to_body = ib
        fib.nuc_site_idx = idx_in_all
        fib.ID = b.ID
        fib.x = axis_s
        fib.v_length = v_growth
        fib.length += fib.v_length * dt
        fibers.append(fib)
        num_nucleated += 1
        site_idcs_nucleating.append(fib.nuc_site_idx)

  return fibers, fibers_gone, bodies, num_nucleated, num_nucleated_imaginary, sites_gone_imaginary, terminal_time_gone_imaginary, site_idcs_nucleating, site_idcs_dying

##############################################################################################
def dynamic_instability_v4(fibers,
                        bodies,
                        prams,
                        options,
                        dt):
  '''
  Dynamic instability of MTs
  '''
  radius_a = prams.periphery_a
  radius_b = prams.periphery_b
  radius_c = prams.periphery_c
  v_growth = prams.v_growth

  for k, fib in enumerate(fibers):
    # Sample a random number

    v_length = v_growth
    if fib.iReachSurface: v_length = v_growth * prams.scale_vg

    if True:
      # otherwise grow with v_growth
      nuc_site_xyz = bodies[fib.attached_to_body].nuc_sites[fib.nuc_site_idx]
      body_radius = bodies[fib.attached_to_body].radius
      if abs(nuc_site_xyz[2]) > body_radius/2 and not fib.stop_growing:
        xfib, yfib, zfib = fib.x[:,0], fib.x[:,1], fib.x[:,2]
        x = xfib[-1]/radius_a
        y = yfib[-1]/radius_b
        z = zfib[-1]/radius_c

        r_true = np.sqrt(xfib[-1]**2 + yfib[-1]**2 + zfib[-1]**2)
        r_fiber = np.sqrt(x**2 + y**2 + z**2)
        phi_fiber = np.arctan2(y, (x + 1e-12))
        theta_fiber = np.arccos(z / (1e-12 + r_fiber))

        x_cort = radius_a * np.sin(theta_fiber) * np.cos(phi_fiber)
        y_cort = radius_b * np.sin(theta_fiber) * np.sin(phi_fiber)
        z_cort = radius_c * np.cos(theta_fiber)

        d2cort = np.sqrt((xfib[-1] - x_cort)**2 + (yfib[-1] - y_cort)**2 + (zfib[-1] - z_cort)**2)

        if d2cort <= 1.5: fib.stop_growing = True

      if fib.stop_growing: v_length = 0

      fib.v_length = v_length
      fib.length_previous = np.copy(fib.length)
      fib.length = fib.length + fib.v_length * dt


  return fibers, bodies
##############################################################################################
def dynamic_instability_v3(fibers,
                        bodies,
                        prams,
                        site_idcs_nucleating,
                        site_idcs_dying,
                        site_idcs_hinged,
                        options,
                        dt):
  '''
  Dynamic instability of MTs
  '''
  radius_a = prams.periphery_a
  radius_b = prams.periphery_b
  radius_c = prams.periphery_c
  v_growth = prams.v_growth
  minL = prams.minL
  fiber_ds = options.fiber_ds

  # First find iReachSurface_fake fibers
  for k, fib in enumerate(fibers):
    if any(fib.nuc_site_idx == site_idcs_hinged):
      fib.iReachSurface_fake = True

    # For a fiber to be hinged, it has to be physically hinged,
    # and also hinged at the random number generator simulation

  temp_fibers = []
  for k, fib in enumerate(fibers):
    # Sample a random number

    v_length = v_growth
    if fib.iReachSurface: v_length = v_growth * prams.scale_vg

    if any(fib.nuc_site_idx == site_idcs_dying): # then fiber goes through CATASTROPHE
      idx_body = fib.attached_to_body
      bodies[idx_body].passive_sites_idcs.append(fib.nuc_site_idx)
      nuc_site_idx_in_active = bodies[idx_body].active_sites_idcs.index(fib.nuc_site_idx)
      del bodies[idx_body].active_sites_idcs[nuc_site_idx_in_active]
    else:
      # otherwise grow with v_growth
      nuc_site_xyz = bodies[fib.attached_to_body].nuc_sites[fib.nuc_site_idx]
      body_radius = bodies[fib.attached_to_body].radius
      if abs(nuc_site_xyz[2]) > body_radius/2 and not fib.stop_growing:
        xfib, yfib, zfib = fib.x[:,0], fib.x[:,1], fib.x[:,2]
        x = xfib[-1]/radius_a
        y = yfib[-1]/radius_b
        z = zfib[-1]/radius_c

        r_true = np.sqrt(xfib[-1]**2 + yfib[-1]**2 + zfib[-1]**2)
        r_fiber = np.sqrt(x**2 + y**2 + z**2)
        phi_fiber = np.arctan2(y, (x + 1e-12))
        theta_fiber = np.arccos(z / (1e-12 + r_fiber))

        x_cort = radius_a * np.sin(theta_fiber) * np.cos(phi_fiber)
        y_cort = radius_b * np.sin(theta_fiber) * np.sin(phi_fiber)
        z_cort = radius_c * np.cos(theta_fiber)

        d2cort = np.sqrt((xfib[-1] - x_cort)**2 + (yfib[-1] - y_cort)**2 + (zfib[-1] - z_cort)**2)

        if d2cort <= 1.5: fib.stop_growing = True

      if fib.stop_growing: v_length = 0

      fib.v_length = v_length
      fib.length_previous = np.copy(fib.length)
      fib.length = fib.length + fib.v_length * dt
      temp_fibers.append(fib)
  # Remove fibers
  fibers = temp_fibers
  # Poisson process to nucleate new MTs
  Ninit = options.num_points

  s = np.linspace(0, 2, Ninit)

  b = bodies[0]
  for idx_in_all in site_idcs_nucleating:
    # Get body's position and orientation
    center, orientation = b.location, b.orientation
    rotation_matrix = orientation.rotation_matrix()

    idx_in_passive = b.passive_sites_idcs.index(idx_in_all)

    ilink = b.nuc_sites[idx_in_all]
    site_location = np.dot(rotation_matrix, ilink)
    site_normal = ilink / np.linalg.norm(ilink)

    del b.passive_sites_idcs[idx_in_passive]
    b.active_sites_idcs.append(idx_in_all)

    # nucleate MT with initial length if minL
    axis = np.dot(rotation_matrix, site_normal)
    axis_s = np.empty((s.size, 3))
    axis_s[:, 0] = axis[0] * s
    axis_s[:, 1] = axis[1] * s
    axis_s[:, 2] = axis[2] * s
    axis_s = axis_s * (minL / 2.0) + site_location + center

    # add new fiber to the fibers list
    fib = fiber.fiber(
        num_points = Ninit,
        fiber_ds = fiber_ds,
        num_points_max=options.num_points_max,
        num_points_finite_diff=options.num_points_finite_diff,
        dt=dt,
        E=prams.Efib,
        length=minL,
        adaptive_num_points=options.adaptive_num_points,
        growing=prams.growing,
        viscosity=prams.viscosity,
        BC_start_0='velocity',
        BC_start_1='angular_velocity')
    fib.force_stall = prams.force_stall
    fib.attached_to_body = 0
    fib.nuc_site_idx = idx_in_all
    fib.ID = b.ID
    fib.x = axis_s
    fib.v_length = v_growth
    fib.length += fib.v_length * dt
    fib.iReachSurface = False
    fib.iReachSurface_fake = False
    fibers.append(fib)

  bodies[0] = b

  return fibers, bodies
##############################################################################################
def dynamic_instability_peripheral(fibers,
                        bodies,
                        prams,
                        options,
                        dt,
                        radius_a,
                        radius_b,
                        radius_c,
                        molecular_motors = None):
  '''
  Dynamic instability of MTs
  '''
  nucleation_rate = prams.nucleation_rate
  rate_catastrophe = prams.rate_catastrophe
  v_growth = prams.v_growth
  fiber_ds = options.fiber_ds
  minL = 0.5

  fibers_gone = []
  # Check if fiber reaches cortex, if so, then it goes through catastrophe
  # Otherwise polymerize, depolymerize
  temp_fibers = []
  site_idcs_dying = []

  for k, fib in enumerate(fibers):
    if fib.attached_to_body is not None: # on the centrosome
      fib.v_length = 0
      temp_fibers.append(fib)
    else: # on the cortex
      if fib.length >= prams.max_length:
        site_idcs_dying.append(fib.nuc_site_idx)
        fibers_gone.append(fibers[k])
        bodies[0].passive_sites_idcs.append(fib.nuc_site_idx)
        nuc_site_idx_in_active = bodies[0].active_sites_idcs.index(fib.nuc_site_idx)
        del bodies[0].active_sites_idcs[nuc_site_idx_in_active]
      else:
        r = np.random.rand(1)
        if r > np.exp(-dt * rate_catastrophe):
          site_idcs_dying.append(fib.nuc_site_idx)
          fibers_gone.append(fibers[k])
          bodies[0].passive_sites_idcs.append(fib.nuc_site_idx)
          nuc_site_idx_in_active = bodies[0].active_sites_idcs.index(fib.nuc_site_idx)
          del bodies[0].active_sites_idcs[nuc_site_idx_in_active]
        else:
          fib.v_length = v_growth
          fib.length_previous = np.copy(fib.length)
          fib.length = fib.length + fib.v_length * dt
          temp_fibers.append(fib)

  fibers = temp_fibers


  # Poisson process to nucleate new MTs
  Ninit = options.num_points

  s = np.linspace(0, 2, Ninit)

  num_nucleated = 0
  site_idcs_nucleating = []
  for ib, b in enumerate(bodies):
    # Put locations of occupied sites into an array
    # Reference config.
    active_sites = np.array([])
    for k in np.arange(len(b.active_sites_idcs)):
      loc = b.nuc_sites[b.active_sites_idcs[k]]
      if active_sites.any():
        active_sites = np.concatenate((active_sites,np.reshape(loc,(1,3))),axis=0)
      else:
        active_sites = np.reshape(loc,(1,3))
    # Find number of MTs to nucleate
    num_to_nucleate = min(np.random.poisson(nucleation_rate * dt), len(b.passive_sites_idcs))
    # Nucleate MTs
    for k in np.arange(num_to_nucleate):
      cannot_place_fiber = True
      ntrial = 0
      while cannot_place_fiber and ntrial < 50:
        # choose a link
        idx_in_passive = np.random.randint(len(b.passive_sites_idcs))
        idx_in_all = b.passive_sites_idcs[idx_in_passive]
        site_location = b.nuc_sites[idx_in_all]
        site_normal = site_location / np.linalg.norm(site_location)

        # if there active links, then check interfilament spacing
        if active_sites.size > 0:
          dummy_links = np.concatenate((active_sites, np.reshape(site_location, (1,3))), axis=0)
          dx = dummy_links[:,0] - dummy_links[:,0,None]
          dy = dummy_links[:,1] - dummy_links[:,1,None]
          dz = dummy_links[:,2] - dummy_links[:,2,None]
          dr = np.sqrt(dx**2 + dy**2 + dz**2)
          dfilament = min(dr[0,1:])
          if dfilament > b.min_ds:
            cannot_place_fiber = False
          else:
            ntrial += 1
        else:
          # if there is no active site, then nucleate anywhere
          cannot_place_fiber = False

      if not cannot_place_fiber:
        # save occupied link location
        if active_sites.size > 0:
          active_sites = np.concatenate((active_sites,np.reshape(site_location,(1,3))), axis=0)
        else:
          active_sites = np.reshape(site_location,(1,3))
        # update active and remaining link lists:
        del b.passive_sites_idcs[idx_in_passive]
        b.active_sites_idcs.append(idx_in_all)

        # nucleate MT with initial length if minL
        axis_s = np.empty((s.size, 3))
        axis_s[:, 0] = site_normal[0] * s
        axis_s[:, 1] = site_normal[1] * s
        axis_s[:, 2] = site_normal[2] * s
        axis_s = axis_s * (minL / 2.0) + site_location

        # add new fiber to the fibers list
        fib = fiber.fiber(
            num_points=Ninit,
            fiber_ds = fiber_ds,
            num_points_max=options.num_points_max,
            num_points_finite_diff=options.num_points_finite_diff,
            dt=dt,
            E=prams.Efib,
            length=minL,
            adaptive_num_points=options.adaptive_num_points,
            growing=prams.growing,
            viscosity=prams.viscosity,
            BC_start_0='force',
            BC_start_1='torque',
            BC_end_0='velocity',
            BC_end_1='torque')
        fib.force_stall = prams.force_stall
        fib.iReachSurface = True
        fib.attached_to_body = None
        fib.nuc_site_idx = idx_in_all
        fib.ID = 'cortex'
        fib.x = axis_s
        fib.hinged_tip = fib.x[-1]
        fib.v_length = v_growth
        fib.length += fib.v_length * dt
        # randomly assing life time from exponential distribution
        fibers.append(fib)
        num_nucleated += 1

  return fibers, fibers_gone, bodies, num_nucleated, site_idcs_nucleating, site_idcs_dying
##############################################################################################
def flow_fibers(f, src, trg, fibers, offset_fibers, eta, integration = 'trapz',
  fib_mats = None, fib_mat_resolutions = None, iupsample = False,
  oseen_fmm = None, fmm_max_pts = 1000):
  '''
  Flow created by fibers at target points.
  '''

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
      weights, weights_up, out3, out4 = fib_mats[indx].get_matrices(fib.length_previous, fib.num_points_up, 'weights_all')
    else:
      weights, out2, out3, out4 = fib_mats[indx].get_matrices(fib.length_previous, fib.num_points_up, 'simpsons')

    P_kerUp, P_kerDn, out3, out4 = fib_mats[indx].get_matrices(fib.length_previous, fib.num_points_up, 'P_kernel')

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

  uprate = fib_mats[0].uprate_poten
  
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
def no_sliding_cortical_pushing(fibers, cortex_a = None, cortex_b = None, cortex_c = None, cortex_radius = None):
  nhinged = 0
  dh = 0.5
  for k, fib in enumerate(fibers):
    if fib.iReachSurface is False and not fib.stop_growing:

      if cortex_radius is not None:
        cortex_a, cortex_b, cortex_c = cortex_radius, cortex_radius, cortex_radius

      if cortex_a is not None:
        xfib, yfib, zfib = fib.x[:,0], fib.x[:,1], fib.x[:,2]
        x = np.copy(xfib[-1])/cortex_a
        y = np.copy(yfib[-1])/cortex_b
        z = np.copy(zfib[-1])/cortex_c

        r_true = np.sqrt(xfib[-1]**2 + yfib[-1]**2 + zfib[-1]**2)
        r_fiber = np.sqrt(x**2 + y**2 + z**2)
        phi_fiber = np.arctan2(y, (x + 1e-12))
        theta_fiber = np.arccos(z / (1e-12 + r_fiber))

        x_cort = cortex_a * np.sin(theta_fiber) * np.cos(phi_fiber)
        y_cort = cortex_b * np.sin(theta_fiber) * np.sin(phi_fiber)
        z_cort = cortex_c * np.cos(theta_fiber)

        dir2fib = np.zeros(3)
        dir2fib[0] = xfib[-1] - x_cort
        dir2fib[1] = yfib[-1] - y_cort
        dir2fib[2] = zfib[-1] - z_cort
        dir2fib = dir2fib / np.linalg.norm(dir2fib)

        d2cort = np.sqrt((xfib[-1] - x_cort)**2 + (yfib[-1] - y_cort)**2 + (zfib[-1] - z_cort)**2)
        cortex_point_r = np.sqrt(x_cort**2 + y_cort**2 + z_cort**2)

        if d2cort <= 0.75:
          fib.iReachSurface = True

    if fib.iReachSurface: nhinged += 1
  print('There are ', nhinged, ' hinged MTs')


  return fibers

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
      weights, weights_up, out3, out4 = fib_mats[indx].get_matrices(fib.length_previous, fib.num_points_up, 'weights_all')
      # upsample force and multiply with weights
      fw[uprate*offset_fibers[k]:uprate*offset_fibers[k+1]] = np.dot(P_kerUp, f[offset_fibers[k]:offset_fibers[k+1]]) * weights_up[:,None]
  else:
    fw = np.zeros((offset_fibers[-1],3))
    for k, fib in enumerate(fibers):
      # Find the index for fib_mats
      indx = np.where(fib_mat_resolutions == fib.num_points)
      indx = indx[0][0]
      if integration == 'trapz':
        weights, weights_up, out3, out4 = fib_mats[indx].get_matrices(fib.length_previous, fib.num_points_up, 'weights_all')
      else:
        weights, out2, out3, out4 = fib_mats[indx].get_matrices(fib.length_previous, fib.num_points_up, 'simpsons')
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

  for offset_fiber, fib in enumerate(fibers):

    if fib.attached_to_body is not None:
      # Get body to which the fiber fib is attached
      k = fib.attached_to_body
      b = bodies[k]

      # Rotation matrix to get current config from reference config
      rotation_matrix = b.orientation.rotation_matrix()
      # Reference location of the nucleating site
      link_loc_ref = b.nuc_sites[fib.nuc_site_idx]

      # Location of link w.r.t. center of mass
      link = np.dot(rotation_matrix, link_loc_ref)
      
      # Find the location of the point in the matrix:
      offset_point = offset_fibers[offset_fiber] * 4 + offset_bodies

      # Rotation matrix to get current config from reference config
      rotation_matrix = b.orientation.rotation_matrix()
      # Reference location of the nucleating site
      link_loc_ref = b.nuc_sites[fib.nuc_site_idx]

      # Location of link w.r.t. center of mass
      link = np.dot(rotation_matrix, link_loc_ref)

      # Find the location of the point in the matrix:
      offset_point = offset_fibers[offset_fiber] * 4 + offset_bodies

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
      As_dok_BC[k*6 + 3, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] += -fib.E * link[1] * D_3[0,:]
      # zy:
      As_dok_BC[k*6 + 3, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] +=  fib.E * link[2] * D_3[0,:]
      # zx:
      As_dok_BC[k*6 + 4, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] += -fib.E * link[2] * D_3[0,:]
      # xz:
      As_dok_BC[k*6 + 4, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] +=  fib.E * link[0] * D_3[0,:]
      # xy:
      As_dok_BC[k*6 + 5, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] += -fib.E * link[0] * D_3[0,:]
      # yx:
      As_dok_BC[k*6 + 5, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] +=  fib.E * link[1] * D_3[0,:]

      # Tension force term:
      # yz - zy:
      As_dok_BC[k*6 + 3, offset_point+fib.num_points*3] += (link[1]*xs[0,2] - link[2]*xs[0,1])
      # zx - xz:
      As_dok_BC[k*6 + 4, offset_point+fib.num_points*3] += (link[2]*xs[0,0] - link[0]*xs[0,2])
      # xy - yx:
      As_dok_BC[k*6 + 5, offset_point+fib.num_points*3] += (link[0]*xs[0,1] - link[1]*xs[0,0])

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
      As_dok_BC[offset_point+fib.num_points*4-14, k*6 + 4] += -link[2]
      # zy:
      As_dok_BC[offset_point+fib.num_points*4-14, k*6 + 5] +=  link[1]
      # zx:
      As_dok_BC[offset_point+fib.num_points*4-13, k*6 + 5] += -link[0]
      # xz:
      As_dok_BC[offset_point+fib.num_points*4-13, k*6 + 3] +=  link[2]
      # xy:
      As_dok_BC[offset_point+fib.num_points*4-12, k*6 + 3] += -link[1]
      # yx:
      As_dok_BC[offset_point+fib.num_points*4-12, k*6 + 4] +=  link[0]

      # Tension equation, left hand side of it (U + Omega x link - \bar{u}_f).xs
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 0] += -xs[0,0]
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 1] += -xs[0,1]
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 2] += -xs[0,2]

      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 3] += (xs[0,1]*link[2] - xs[0,2]*link[1])
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 4] += (xs[0,2]*link[0] - xs[0,0]*link[2])
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 5] += (xs[0,0]*link[1] - xs[0,1]*link[0])


      # Matrix A_fiber_body, for angle
      # Clamped boundary condition: dXs/dt = Omega x Xs or Omega x link_direction
      link_norm = np.sqrt(link[0]**2 + link[1]**2 + link[2]**2)
      link_dir = link / link_norm
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
def get_link_force_torque(bodies,fib,fib_mats,fib_mat_resolutions):
  '''
  Building link matrix (fibers' boundary conditions)
  '''
  # Get body to which the fiber fib is attached
  k = fib.attached_to_body
  b = bodies[k]

  # Rotation matrix to get current config from reference config
  rotation_matrix = b.orientation.rotation_matrix()
  # Reference location of the nucleating site
  link_loc_ref = b.nuc_sites[fib.nuc_site_idx]

  # Location of link w.r.t. center of mass
  link = np.dot(rotation_matrix, link_loc_ref)

  # Find the index for fib_mats
  indx = np.where(fib_mat_resolutions == fib.num_points)
  indx = indx[0][0]

  # Get the class that has the matrices
  fib_mat = fib_mats[indx]
  out1, D_2, D_3, out4 = fib_mat.get_matrices(fib.length, fib.num_points_up, 'Ds')
  xs = fib.xs

  force = np.zeros((1,3))
  torque = np.zeros((1,3))
  # Rectangular mathod, Driscoll and Hale
  # Matrix A_body_fiber, for position
  # Force by fiber on body at s = 0, Fext = -F|s=0 = -(EXsss - TXs)
  # Bending term:
  xn_sss = np.dot(D_3, fib.x_new)
  xn_ss = np.dot(D_2, fib.x_new)
  force[0,0] +=  -fib.E * xn_sss[0,0]
  force[0,1] +=  -fib.E * xn_sss[0,1]
  force[0,2] +=  -fib.E * xn_sss[0,2]
  # Tension term:
  force[0,0] += xs[0,0]*fib.tension_new[0]
  force[0,1] += xs[0,1]*fib.tension_new[0]
  force[0,2] += xs[0,2]*fib.tension_new[0]

  # Torque by fiber on body at s = 0, Lext = (L + link_loc x F) = -(E(Xss x Xs) + link_loc x (EXsss - TXs))
  # Bending force term:
  # yz:
  torque[0,0] += -fib.E * link[1] * xn_sss[0,2]
  # zy:
  torque[0,0] +=  fib.E * link[2] * xn_sss[0,1]
  # zx:
  torque[0,1] += -fib.E * link[2] * xn_sss[0,0]
  # xz:
  torque[0,1] +=  fib.E * link[0] * xn_sss[0,2]
  # xy:
  torque[0,2] += -fib.E * link[0] * xn_sss[0,1]
  # yx:
  torque[0,2] +=  fib.E * link[1] * xn_sss[0,0]

  # Tension force term:
  # yz - zy:
  torque[0,0] += (link[1]*xs[0,2] - link[2]*xs[0,1])*fib.tension_new[0]
  # zx - xz:
  torque[0,1] += (link[2]*xs[0,0] - link[0]*xs[0,2])*fib.tension_new[0]
  # xy - yx:
  torque[0,2] += (link[0]*xs[0,1] - link[1]*xs[0,0])*fib.tension_new[0]

  # Fiber torque (L):
  # yz:
  torque[0,0] += -fib.E * xs[0,2] * xn_ss[0,1]
  # zy:
  torque[0,0] +=  fib.E * xs[0,1] * xn_ss[0,2]
  # zx:
  torque[0,1] += -fib.E * xs[0,0] * xn_ss[0,2]
  # xz:
  torque[0,1] +=  fib.E * xs[0,2] * xn_ss[0,0]
  # xy:
  torque[0,2] += -fib.E * xs[0,1] * xn_ss[0,0]
  # yx:
  torque[0,2] +=  fib.E * xs[0,0] * xn_ss[0,1]



  return force, torque
##############################################################################################
def build_hard_link_matrix(bodies,fibers,offset_fibers,offset_bodies,fib_mats,fib_mat_resolutions):
  '''
  Building link matrix (fibers' boundary conditions)
  '''
  system_size = 4*offset_fibers[-1] + 6
  As_dok_BC = scsp.dok_matrix((system_size, system_size))
  center_of_mass = np.zeros(3)
  for k, b in enumerate(bodies):
    center_of_mass += b.location
  center_of_mass = center_of_mass / len(bodies)

  for k, b in enumerate(bodies):
    # Get links location
    r2cent = np.linalg.norm(center_of_mass - b.location)
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

      if b.location[2] - center_of_mass[2] > 0: # TOP CENT
        links_loc[i,2] += r2cent
      else:
        links_loc[i,2] -= r2cent

      # Rectangular mathod, Driscoll and Hale
      # Matrix A_body_fiber, for position
      # Force by fiber on body at s = 0, Fext = -F|s=0 = -(EXsss - TXs)
      # Bending term:
      As_dok_BC[0, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] +=  -fib.E * D_3[0,:]
      As_dok_BC[1, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] +=  -fib.E * D_3[0,:]
      As_dok_BC[2, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] +=  -fib.E * D_3[0,:]
      # Tension term:
      As_dok_BC[0, offset_point+fib.num_points*3] += xs[0,0]
      As_dok_BC[1, offset_point+fib.num_points*3] += xs[0,1]
      As_dok_BC[2, offset_point+fib.num_points*3] += xs[0,2]

      # Torque by fiber on body at s = 0, Lext = (L + link_loc x F) = -(E(Xss x Xs) + link_loc x (EXsss - TXs))
      # Bending force term:
      # yz:
      As_dok_BC[3, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] += -fib.E * links_loc[i,1] * D_3[0,:]
      # zy:
      As_dok_BC[3, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] +=  fib.E * links_loc[i,2] * D_3[0,:]
      # zx:
      As_dok_BC[4, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] += -fib.E * links_loc[i,2] * D_3[0,:]
      # xz:
      As_dok_BC[4, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] +=  fib.E * links_loc[i,0] * D_3[0,:]
      # xy:
      As_dok_BC[5, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] += -fib.E * links_loc[i,0] * D_3[0,:]
      # yx:
      As_dok_BC[5, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] +=  fib.E * links_loc[i,1] * D_3[0,:]

      # Tension force term:
      # yz - zy:
      As_dok_BC[3, offset_point+fib.num_points*3] += (links_loc[i,1]*xs[0,2] - links_loc[i,2]*xs[0,1])
      # zx - xz:
      As_dok_BC[4, offset_point+fib.num_points*3] += (links_loc[i,2]*xs[0,0] - links_loc[i,0]*xs[0,2])
      # xy - yx:
      As_dok_BC[5, offset_point+fib.num_points*3] += (links_loc[i,0]*xs[0,1] - links_loc[i,1]*xs[0,0])

      # Fiber torque (L):
      # yz:
      As_dok_BC[3, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] += -fib.E * xs[0,2] * D_2[0,:]
      # zy:
      As_dok_BC[3, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] +=  fib.E * xs[0,1] * D_2[0,:]
      # zx:
      As_dok_BC[4, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] += -fib.E * xs[0,0] * D_2[0,:]
      # xz:
      As_dok_BC[4, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] +=  fib.E * xs[0,2] * D_2[0,:]
      # xy:
      As_dok_BC[5, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] += -fib.E * xs[0,1] * D_2[0,:]
      # yx:
      As_dok_BC[5, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] +=  fib.E * xs[0,0] * D_2[0,:]


      # Matrix A_fiber_body, for position
      # dx/dt = U + Omega x link_loc (move to LHS so -U -Omega x link_loc)
      # Linear velocity part (U)
      As_dok_BC[offset_point+fib.num_points*4-14, 0] += -1.0
      As_dok_BC[offset_point+fib.num_points*4-13, 1] += -1.0
      As_dok_BC[offset_point+fib.num_points*4-12, 2] += -1.0
      # Angular velocity part (Omega)
      # yz:
      As_dok_BC[offset_point+fib.num_points*4-14, 4] += -links_loc[i,2]
      # zy:
      As_dok_BC[offset_point+fib.num_points*4-14, 5] +=  links_loc[i,1]
      # zx:
      As_dok_BC[offset_point+fib.num_points*4-13, 5] += -links_loc[i,0]
      # xz:
      As_dok_BC[offset_point+fib.num_points*4-13, 3] +=  links_loc[i,2]
      # xy:
      As_dok_BC[offset_point+fib.num_points*4-12, 3] += -links_loc[i,1]
      # yx:
      As_dok_BC[offset_point+fib.num_points*4-12, 4] +=  links_loc[i,0]

      # Tension equation, left hand side of it (U + Omega x link - \bar{u}_f).xs
      As_dok_BC[offset_point+fib.num_points*4-11, 0] += -xs[0,0]
      As_dok_BC[offset_point+fib.num_points*4-11, 1] += -xs[0,1]
      As_dok_BC[offset_point+fib.num_points*4-11, 2] += -xs[0,2]

      As_dok_BC[offset_point+fib.num_points*4-11, 3] += (xs[0,1]*links_loc[i,2] - xs[0,2]*links_loc[i,1])
      As_dok_BC[offset_point+fib.num_points*4-11, 4] += (xs[0,2]*links_loc[i,0] - xs[0,0]*links_loc[i,2])
      As_dok_BC[offset_point+fib.num_points*4-11, 5] += (xs[0,0]*links_loc[i,1] - xs[0,1]*links_loc[i,0])


      # Matrix A_fiber_body, for angle
      # Clamped boundary condition: dXs/dt = Omega x Xs or Omega x link_direction
      link_norm = np.sqrt(links_loc[i,0]**2 + links_loc[i,1]**2 + links_loc[i,2]**2)
      link_dir = links_loc[i,:] / link_norm
      # yz:
      As_dok_BC[offset_point+fib.num_points*4-10, 4] += -link_dir[2]
      # zy:
      As_dok_BC[offset_point+fib.num_points*4-10, 5] +=  link_dir[1]
      # zx:
      As_dok_BC[offset_point+fib.num_points*4-9, 5]  += -link_dir[0]
      # xz:
      As_dok_BC[offset_point+fib.num_points*4-9, 3]  +=  link_dir[2]
      # xy:
      As_dok_BC[offset_point+fib.num_points*4-8, 3]  += -link_dir[1]
      # yx:
      As_dok_BC[offset_point+fib.num_points*4-8, 4]  +=  link_dir[0]


  return scsp.csr_matrix(As_dok_BC)

  ##############################################################################################

def get_fibers_and_bodies_matrices(fibers, bodies, shell, system_size,
  offset_fibers, offset_bodies, force_fibers, motor_force_fibers, force_bodies, v_on_fibers, v_on_bodies, v_on_shell,
  fib_mats, fib_mat_resolutions, inextensibility = 'penalty', velAt0 = None, velAt1 = None, ihydro = False):

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
    BC_start_vec_0 = np.zeros(3)
    BC_start_vec_1 = np.zeros(3)
    BC_end_vec_1 = np.zeros(3)
    BC_end_vec_0 = np.zeros(3)
    if fib.BC_start_0 is 'force':
      if force_on is not None:
        BC_start_vec_0 = force_on[0]
    if fib.BC_end_0 is 'force':
      if force_on is not None:
        BC_end_vec_0 = force_on[-1]
    if velAt0 is not None:
      BC_start_vec_0 = velAt0
    if velAt1 is not None:
      BC_start_vec_1 = velAt1
    fib.set_BC(BC_start_0=fib.BC_start_0,
      BC_start_1=fib.BC_start_1,
      BC_end_0=fib.BC_end_0,
      BC_end_1 = fib.BC_end_1,
      BC_end_vec_0=BC_end_vec_0,
      BC_end_vec_1 = BC_end_vec_1,
      BC_start_vec_0 = BC_start_vec_0,
      BC_start_vec_1 = BC_start_vec_1)

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
