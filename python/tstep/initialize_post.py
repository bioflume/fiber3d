# 2. READ INPUT FILE AND SET PARAMETERS AND OPTIONS
# 2.1. If there is no input file, then generate fibers with given length, position
# 3. GENERATE FIBERS, BODIES AND BOUNDARIES
# 4. SET SPATIAL RESOLUTION
# 6. INITIALIZE OUTPUT FILES
# 7. OUTPUT PARAMETERS, OPTIONS AND FILE IDS TO WRITE

from __future__ import print_function
import numpy as np
import sys
import scipy.linalg as scla
import scipy.sparse as scsp
import scipy.sparse.linalg as scspla
from scipy.interpolate import interp1d
import argparse
import subprocess
import time
import stkfmm
import os
from scipy.spatial import ConvexHull
try:
  import cPickle as cpickle
except:
  try:
    import cpickle
  except:
    import _pickle as cpickle


# Find project functions
found_functions = False
path_to_append = ''
while found_functions is False:
  try:
    from read_input import read_links_file
    from utils import timer
    from utils import barycentricMatrix as bary
    from fiber import fiber
    from body import body
    from quaternion import quaternion
    from shape_gallery import shape_gallery
    from quadratures import Smooth_Closed_Surface_Quadrature_RBF
    from periphery import periphery
    from tstep import time_step_container

    found_functions = True
  except ImportError:
    path_to_append += '../'
    print('searching functions in path ', path_to_append)
    sys.path.append(path_to_append)
    if len(path_to_append) > 21:
      print('\nProjected functions not found. Edit path in multi_bodies.py')
      sys.exit()

class set_options(object):
  def __init__(self,
              repulsion = False,
              adaptive_num_points = False,
              filtering = False,
              dt = 1e-2,
              tol_gmres = 1e-10,
              fiber_ds = 0.5/32,
              num_points = 32,
              num_points_finite_diff = 4,
              num_points_max = 256,
              random_seed = None,
              uprate = 4,
              inonlocal = False,
              output_name = 'defaultOutput',
              n_save = 1,
              verbose = False,
              precompute_body_PC = False,
              useFMM = False,
              fmm_order = 8,
              fmm_max_pts = 1000,
              body_quadrature_radius = None,
              Nblobs = None,
              Nperiphery = None,
              integration = 'trapz',
              iupsample = False,
              inextensibility = 'penalty',
              penalty_param = 500,
              iExternalForce = False,
              iCytoPulling = False,
              iNoSliding = False,
              dynInstability = False,
              time_step_scheme = 'time_step_hydro'):
    # RUN OPTIONS (~ALGORITHMS)
    self.repulsion = repulsion
    self.time_step_scheme = time_step_scheme
    self.filtering = filtering
    self.dt = dt
    self.tol_gmres = tol_gmres
    self.fiber_ds = fiber_ds
    self.adaptive_num_points = adaptive_num_points
    self.num_points = num_points
    self.num_points_finite_diff = num_points_finite_diff
    self.num_points_max = num_points_max
    self.random_seed = random_seed
    self.uprate = uprate
    self.inonlocal = inonlocal
    self.output_name = output_name
    self.n_save = n_save
    self.verbose = verbose
    self.precompute_body_PC = precompute_body_PC
    self.useFMM = useFMM
    self.fmm_order = fmm_order
    self.fmm_max_pts = fmm_max_pts
    self.body_quadrature_radius = body_quadrature_radius
    self.Nblobs = Nblobs
    self.Nperiphery = Nperiphery
    self.integration = integration
    self.iupsample = iupsample
    self.inextensibility = inextensibility
    self.penalty_param = penalty_param
    self.iExternalForce = iExternalForce
    self.iCytoPulling = iCytoPulling
    self.iNoSliding = iNoSliding
    self.dynInstability = dynInstability

class set_parameters(object):
  def __init__(self,
              bodies_file = None,
              fibers_file = None,
              time_step_file = None,
              nskip_steps = None,
              eta = 1.0,
              epsilon = 1e-03,
              Efib = None,
              periphery = None,
              periphery_radius = None,
              periphery_a = None,
              periphery_b = None,
              periphery_c = None,
              mt_surface_radius = None,
              mt_surface_radius_a = None,
              mt_surface_radius_b = None,
              mt_surface_radius_c = None,
              MT_length = None,
              motor_strength_gradient_function = None,
              motor_strength_gradient_along_axis = None,
              motor_strength_gradient_regions = None,
              motor_strength_gradient = None,
              xrange = None,
              yrange = None,
              zrange = None):
    # FLOW PARAMETERS
    self.nskip_steps = nskip_steps
    self.bodies_file = bodies_file
    self.nskip_steps = nskip_steps
    self.fibers_file = fibers_file
    self.time_step_file = time_step_file
    self.eta = eta
    self.epsilon = epsilon
    self.Efib = Efib
    self.periphery = periphery
    self.periphery_radius = periphery_radius
    self.periphery_a = periphery_a
    self.periphery_b = periphery_b
    self.periphery_c = periphery_c
    self.mt_surface_radius = mt_surface_radius
    self.mt_surface_radius_a = mt_surface_radius_a
    self.mt_surface_radius_b = mt_surface_radius_b
    self.mt_surface_radius_c = mt_surface_radius_c
    self.MT_length = MT_length
    self.motor_strength_gradient_function = motor_strength_gradient_function
    self.motor_strength_gradient_along_axis = motor_strength_gradient_along_axis
    self.motor_strength_gradient_regions = motor_strength_gradient_regions
    self.motor_strength_gradient = motor_strength_gradient
    self.xrange = xrange
    self.yrange = yrange
    self.zrange = zrange
    

def initialize_from_file(options,prams):
  '''
  Read data from input files, initialize fibers, bodies and molecular motors
  '''


  
  # Create body_fibers
  time_steps = []

  # Read all the data from different checkpoints
  for idFile, time_file in enumerate(prams.time_step_file):
    print('Reading', time_file)
    # First read the time files
    dt_time_nsteps = np.loadtxt(time_file, delimiter = ' ')
    dt = dt_time_nsteps[0,0]
    time = dt_time_nsteps[:,1]
    nsteps = dt_time_nsteps[:,2]
    if idFile == 0:
      time_all = time
      nsteps_all = nsteps
      nstep_last = 0
    else:
      nstep_last = nsteps_all[-1]
      time_all = np.concatenate((time_all, time_all[-1] + time[1:]))
      nsteps_all = np.concatenate((nsteps_all, nsteps_all[-1] + nsteps[1:]))


    # Read the fibers and the bodies
    print('Reading', prams.fibers_file[idFile])
    fiber_data = np.loadtxt(prams.fibers_file[idFile], delimiter = ' ')
    offset = 0
    step_idcs = np.int32(nsteps / options.n_save)
    for k in step_idcs:
      fibers = []
      bodies = []
      # Load the fibers
      nfibers = int(fiber_data[offset,0])
      for ifib in np.arange(nfibers):
        Nfib = int(fiber_data[offset+1, 0])
        Lfib = fiber_data[offset+1, 2]
        xyz_fib = fiber_data[offset + 2: offset+2+Nfib,:3]
        if xyz_fib.size//3 == Nfib:
          ifiber = fiber.fiber(num_points = Nfib,
                               num_points_max = options.num_points_max,
                               num_points_finite_diff = options.num_points_finite_diff,
                               dt = options.dt,
                               E = prams.Efib,
                               length = Lfib,
                               epsilon = prams.epsilon,
                               inonlocal = False,
                               ireparam = False,
                               adaptive_num_points = False,
                               tstep_order = 1,
                               viscosity = prams.eta)
          ifiber.ID = 'fibers'
          ifiber.x = xyz_fib
          ifiber.length_0 = 0.5
          ifiber.num_points_0 = options.num_points
          if idFile == 0:
            fibers.append(ifiber)
          else:
            if k > 0: fibers.append(ifiber)
        offset = offset + Nfib + 1
      offset = offset + 1
      iSaveStep = False
      if idFile == 0:
        iSaveStep = True
      else:
        if k > 0: iSaveStep = True

      if iSaveStep:
        time_step = time_step_container.time_step_container(prams = prams,
                                                            options = options,
                                                            fibers = fibers,
                                                            bodies = bodies,
                                                            tstep = int(nstep_last + nsteps[k]))
        time_steps.append(time_step)


  return time_steps, time_all, nsteps_all
