
# 2. READ INPUT FILE AND SET PARAMETERS AND OPTIONS
# 2.1. If there is no input file, then generate fibers with given length, position
# 3. GENERATE FIBERS, BODIES AND BOUNDARIES
# 4. SET SPATIAL RESOLUTION
# 6. INITIALIZE OUTPUT FILES
# 7. OUTPUT PARAMETERS, OPTIONS AND FILE IDS TO WRITE

from __future__ import print_function
import numpy as np
import sys
import argparse
import subprocess
import time
from scipy.interpolate import interp1d
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
    from read_input import read_input
    from read_input import read_fibers_file
    from read_input import read_vertex_file
    from read_input import read_clones_file
    from read_input import read_links_file
    from utils import timer
    from utils import nonlinear
    from utils import cheb
    from utils import barycentricMatrix as bary
    from fiber import fiber
    from kernels import kernels
    from body import body
    from molecular_motor import molecular_motor
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
    adaptive_num_points = False,
    ireparam = False,
    reparam_iter = 10,
    reparam_degree = 6,
    repulsion = False,
    filtering = False,
    adaptive_time = False,
    time_step_scheme = 'fiber_hydro_implicit_time_step',
    order = 1,
    dt = 1e-3,
    dt_min = 1e-4,
    dt_max = 0.1,
    tol_tstep = 1e-2,
    tol_gmres = 1e-12,
    num_points = 32,
    num_points_finite_diff = 4,
    num_points_max = 96,
    random_seed = 1,
    inonlocal = False,
    output_txt_files = True,
    output_name = 'defaultOutput',
    n_save = 1,
    verbose = False,
    precompute_body_PC = False,
    isaveForces = False,
    iFixObjects = False,
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
    irelease = False,
    release_check = 'time',
    release_condition = 1000.0,
    iExternalForce = False,
    iCytoPulling = False,
    slipVelOn = False,
    cytoPull_Elongation = False,
    iDynInstability = False,
    iFiberOnly = True,
    belowBifur = False,
    cortRatio = 0):
    # RUN OPTIONS (~ALGORITHMS)
    self.adaptive_num_points = adaptive_num_points
    self.ireparam = ireparam
    self.num_points = num_points
    self.num_points_finite_diff = num_points_finite_diff
    self.num_points_max = num_points_max
    self.reparam_iter = reparam_iter
    self.reparam_degree = reparam_degree
    self.filtering = filtering
    self.adaptive_time = adaptive_time
    self.time_step_scheme = time_step_scheme
    self.order = order
    self.dt = dt
    self.dt_max = dt_max
    self.dt_min = dt_min
    self.tol_tstep = tol_tstep
    self.tol_gmres = tol_gmres
    self.num_points = num_points
    self.inonlocal = inonlocal
    self.output_name = output_name
    self.n_save = n_save
    self.verbose = verbose
    self.random_seed = random_seed
    self.output_txt_files = output_txt_files
    self.repulsion = repulsion
    self.precompute_body_PC = precompute_body_PC
    self.isaveForces = isaveForces
    self.iFixObjects = iFixObjects # flag to fix or not the objects
    self.useFMM = useFMM # flag to use FMM or not
    self.fmm_order = fmm_order
    self.fmm_max_pts = fmm_max_pts
    self.Nblobs = Nblobs # # of points on rigid bodies
    self.Nperiphery = Nperiphery # # of points on periphery
    self.body_quadrature_radius = body_quadrature_radius # body quadrature radius
    self.integration = integration # trapezoidal rule for integration or 'simpsons'
    self.iupsample = iupsample
    self.inextensibility = inextensibility # either penalty or lagrange_multi
    self.penalty_param = penalty_param
    self.release_check = release_check
    self.release_condition = release_condition
    self.irelease = irelease
    self.iExternalForce = iExternalForce
    self.iCytoPulling = iCytoPulling
    self.slipVelOn = slipVelOn
    self.cytoPull_Elongation = cytoPull_Elongation
    self.iDynInstability = iDynInstability
    self.iFiberOnly = iFiberOnly
    self.belowBifur = belowBifur
    self.cortRatio = cortRatio
class set_parameters(object):
  def __init__(self, eta = 1.0,
               epsilon = 1e-03,
               final_time = 1.0,
               Efib = None,
               nucleus_radius = [],
               nucleus_position = [],
               periphery = None,
               periphery_radius = None,
               periphery_a = None,
               periphery_b = None,
               periphery_c = None,
               len_nuc2fib = 5e-02,
               len_nuc2bdy = 1e-01,
               len_nuc2nuc = 3e-01,
               scale_nuc2fib = 1.0,
               scale_nuc2bdy = 1.0,
               scale_nuc2nuc = 250.0,
               cortex_radius = None,
               fiber_body_attached = False,
               growing = 0,
               motor_left = 0.8,
               motor_right = 0.8,
               only_body_radius = None,
               iEllipsoid = False,
               MT_length = 1,
               Nmts_gradient_along_axis = None,
               Nmts_regions = None,
               Nmts_in_each_region = None,
               motor_strength_gradient_function = None,
               motor_strength_gradient_along_axis = None,
               motor_strength_gradient_regions = None,
               motor_strength_gradient = None,
               mt_surface_radius = None,
               mt_surface_radius_a = None,
               mt_surface_radius_b = None,
               mt_surface_radius_c = None,
               ncompute_vel = 500,
               xrange = None,
               yrange = None,
               zrange = None,
               fiber_resume = None,
               rotlet_off_time = None,
               rotlet_torque = None):
    # FLOW PARAMETERS
    self.eta = eta # fluid viscosity
    self.epsilon = epsilon # for self-mobility terms
    self.final_time = final_time # time horizon
    self.nucleus_radius = nucleus_radius # when mimicking nucleus with a source of repulsion
    self.nucleus_position = nucleus_position # position of such nucleus
    self.periphery = periphery # geometry of peripher (sphere, ellipsoid)
    self.periphery_radius = periphery_radius # radius of periphery
    self.cortex_radius = cortex_radius
    self.Efib = Efib # Young's modulus of fibers (one for all, or give them separately in input file)
    self.scale_nuc2fib = scale_nuc2fib # repulsion strength for nucleus-fiber interaction
    self.scale_nuc2bdy = scale_nuc2bdy # repulsion strength for nucleus-body interaction
    self.scale_nuc2nuc = scale_nuc2nuc # repulsion strength for nucleus-nucleus interaction
    self.len_nuc2fib = len_nuc2fib # repulsion length scale for nucleus-fiber interaction
    self.len_nuc2bdy = len_nuc2bdy # repulsion length scale for nucleus-body interaction
    self.len_nuc2nuc = len_nuc2nuc # repulsion length scale for nucleus-nucleus interaction
    self.fiber_body_attached = fiber_body_attached # flag for a simulation with fiber-body attached
    self.periphery_a = periphery_a
    self.periphery_b = periphery_b
    self.periphery_c = periphery_c
    self.growing = growing
    self.motor_left = motor_left
    self.motor_right = motor_right
    self.only_body_radius = only_body_radius
    self.iEllipsoid = iEllipsoid
    self.MT_length = MT_length
    self.Nmts_gradient_along_axis = Nmts_gradient_along_axis
    self.Nmts_regions = Nmts_regions
    self.Nmts_in_each_region = Nmts_in_each_region
    self.motor_strength_gradient_function = motor_strength_gradient_function
    self.motor_strength_gradient_along_axis = motor_strength_gradient_along_axis
    self.motor_strength_gradient_regions = motor_strength_gradient_regions
    self.motor_strength_gradient = motor_strength_gradient
    self.mt_surface_radius = mt_surface_radius
    self.mt_surface_radius_a = mt_surface_radius_a
    self.mt_surface_radius_b = mt_surface_radius_b
    self.mt_surface_radius_c = mt_surface_radius_c
    self.ncompute_vel = ncompute_vel
    self.xrange = xrange
    self.yrange = yrange
    self.zrange = zrange
    self.fiber_resume = fiber_resume
    self.rotlet_off_time = rotlet_off_time
    self.rotlet_torque = rotlet_torque

def initialize_from_file(input_file,options,prams):
  '''
  Read data from input files, initialize fibers, bodies and molecular motors
  '''

  # Set random generator state
  np.random.seed(int(options.random_seed))
  # Save random generator state
  with open(options.output_name + '.random_state', 'wb') as f:
    cpickle.dump(np.random.get_state(), f)

  # Save options and parameters of simulation
  with open(options.output_name + '_options_parameters.info', 'w') as f:
    f.write('adaptive_num_points  ' + str(options.adaptive_num_points) + '\n')
    f.write('num_points           ' + str(options.num_points) + '\n')
    f.write('num_points_finit_diff           ' + str(options.num_points_finite_diff) + '\n')
    f.write('num_points_max       ' + str(options.num_points_max) + '\n')
    f.write('reparameterization   ' + str(options.ireparam) + '\n')
    f.write('filtering            ' + str(options.filtering) + '\n')
    f.write('adaptive_time        ' + str(options.adaptive_time) + '\n')
    f.write('time_step_order      ' + str(options.order) + '\n')
    f.write('time_step_tolerance  ' + str(options.tol_tstep) + '\n')
    f.write('GMRES_tolerance      ' + str(options.tol_gmres) + '\n')
    f.write('Nonlocal inters.     ' + str(options.inonlocal) + '\n')
    f.write('Viscosity(eta)       ' + str(prams.eta) + '\n')
    f.write('Final time           ' + str(prams.final_time) + '\n')
    f.write('Epsilon in mobility  ' + str(prams.epsilon) + '\n')
    f.write('Penalty parameter    ' + str(options.penalty_param) + '\n')
    if options.Nperiphery is not None:
      f.write('Number of nodes on periphery ' + str(options.Nperiphery) + '\n')
    if options.repulsion is True:
      f.write('Repulsion is on \n')

  # Create body_fibers
  bodies = []
  bodies_types = []
  bodies_names = []
  fibers = []
  fibers_types = []
  fibers_names = []
  MM = []
  f_fibers_ID = []
  f_fibers_forces_ID = []
  f_bodies_ID = []
  f_bodies_forces_ID = []
  f_molecular_motors_ID = []
  xyz_sites = []

  MM_on_moving_surf = []
  bodies_mm_attached = []
  bodies_types_mm_attached = []
  bodies_names_mm_attached = []
  f_bodies_mm_attached_ID = []
  f_bodies_mm_attached_forces_ID = []
  f_mm_on_moving_surf_ID = []

  if prams.periphery is 'sphere':
    radius_a = prams.mt_surface_radius
    radius_b = prams.mt_surface_radius
    radius_c = prams.mt_surface_radius
  elif prams.periphery is 'ellipsoid':
    radius_a = prams.mt_surface_radius_a
    radius_b = prams.mt_surface_radius_b
    radius_c = prams.mt_surface_radius_c

  min_ds, Nsites = 0.1, 0
  s = np.linspace(0, 2, options.num_points)

  print('Generating the MTs on a cortex')
  if prams.periphery is 'sphere':
    for idx, mt_region in enumerate(prams.Nmts_regions):
      print('Generating MTs for the region: ', mt_region, ' in ', prams.Nmts_gradient_along_axis)
      Nmts_to_gen = prams.Nmts_in_each_region[idx]
      print(Nmts_to_gen, ' will be generated.')
      MTs_gen, ntrials = 0, 0
      while MTs_gen < Nmts_to_gen and ntrials <= 100:
        # Sample a point
        xq2 = radius_a**2 * np.random.randn()
        yq2 = radius_b**2 * np.random.randn()
        zq2 = radius_c**2 * np.random.randn()

        # Project it onto the surface
        d = np.sqrt(xq2**2 / radius_a**2 + yq2**2 / radius_b**2 + zq2**2 / radius_c**2)
        xq = xq2 / d
        yq = yq2 / d
        zq = zq2 / d
        ilink = np.array([xq, yq, zq])

        iPlace = False
        if prams.Nmts_gradient_along_axis is 'x':
          check_loc = xq
        elif prams.Nmts_gradient_along_axis is 'y':
          check_loc = yq
        elif prams.Nmts_gradient_along_axis is 'z':
          check_loc = zq
        
        if prams.Nmts_gradient_along_axis is 'uniform':
          iPlace = True
        else:
          if idx == 0:
            if check_loc >= mt_region[0] and check_loc <= mt_region[1]: iPlace = True
          else:
            if check_loc > mt_region[0] and check_loc <= mt_region[1]: iPlace = True
        
        if prams.Nmts_gradient_along_axis is 'uniform': iPlace = True

        if iPlace:
          if Nsites == 0:
            xyz_sites = np.reshape(ilink, (1,3))
            Nsites += 1
          else:
            dummy = np.concatenate((xyz_sites, np.reshape(ilink, (1,3))), axis = 0)
            dx = dummy[:,0] - dummy[:,0,None]
            dy = dummy[:,1] - dummy[:,1,None]
            dz = dummy[:,2] - dummy[:,2,None]
            dr = np.sqrt(dx**2 + dy**2 + dz**2)
            dfilament = min(dr[0,1:])
            if dfilament >= min_ds:
              xyz_sites = np.copy(dummy)
              Nsites += 1
            else:
              iPlace = False
              ntrials += 1
              if ntrials >= 100:
                print('Too many trials to generate MTs!!, STOP!')
                input()

        if iPlace: # Create a fiber
          MTs_gen += 1
          axis_s = np.empty((s.size,3))
          normal = -ilink / np.linalg.norm(ilink)
          axis_s[:,0] = normal[0] * s
          axis_s[:,1] = normal[1] * s
          axis_s[:,2] = normal[2] * s
          axis_s = axis_s * prams.MT_length / 2 + ilink

          fib = fiber.fiber(num_points = options.num_points,
                            num_points_max = options.num_points_max,
                            num_points_finite_diff = options.num_points_finite_diff,
                            dt = options.dt,
                            E = prams.Efib,
                            length = prams.MT_length,
                            adaptive_num_points = options.adaptive_num_points,
                            viscosity = prams.eta)
          fib.ID = 'fibers'
          fib.x = axis_s
          fibers.append(fib)
  elif prams.periphery is 'ellipsoid' and prams.Nmts_gradient_along_axis is not None:
    def ellipsoid(t, u, a=radius_a, b=radius_b, c=radius_c):
      return np.array([a*np.sin(u)*np.cos(t), b*np.sin(u)*np.sin(t), c*np.cos(u)]) 
    
    Nmts_to_gen = np.sum(prams.Nmts_in_each_region)
    t, u = np.meshgrid(np.linspace(0, 2*np.pi, 25), np.linspace(0, np.pi, 25))
    coords = ellipsoid(t, u)
    # Surface cumulator
    delta_t_temp = np.diff(coords, axis=2)
    delta_u_temp = np.diff(coords, axis=1)

    delta_t = np.zeros(coords.shape)
    delta_u = np.zeros(coords.shape)

    delta_t[:coords.shape[0], :coords.shape[1], 1:coords.shape[2]] = delta_t_temp
    delta_u[:coords.shape[0], 1:coords.shape[1], :coords.shape[2]] = delta_u_temp

    delta_S = np.linalg.norm(np.cross(delta_t, delta_u, 0, 0), axis=2)

    cum_S_t = np.cumsum(delta_S.sum(axis=0))
    cum_S_u = np.cumsum(delta_S.sum(axis=1))

    # r_surface_from_data
    rand_S_t = np.random.rand(5*Nmts_to_gen) * cum_S_t[-1]
    rand_S_u = np.random.rand(5*Nmts_to_gen) * cum_S_u[-1]
    rand_t = interp1d(cum_S_t, t[0, :])(rand_S_t)
    rand_u = interp1d(cum_S_u, u[:, 0])(rand_S_u)

    rand_coords = ellipsoid(rand_t, rand_u)
    rand_coords = rand_coords.transpose()
    xyz_sites = np.reshape(rand_coords[0],(1,3))
    Nsites = 1
    MTs_gen = 1
    xyz_sites_true = np.array([])
    print('Nmts_to_gen: ', Nmts_to_gen)
    for idx in np.arange(5*Nmts_to_gen):
      print('Generating MTs')
      x = np.reshape(rand_coords[idx],(1,3))
      dummy = np.concatenate((xyz_sites,x),axis = 0)
      dx = dummy[:,0] - dummy[:,0,None]
      dy = dummy[:,1] - dummy[:,1,None]
      dz = dummy[:,2] - dummy[:,2,None]

      dr = np.sqrt(dx**2 + dy**2 + dz**2)
      dfilament = min(dr[0,1:])
      print('dfilament: ', dfilament)
      if dfilament > min_ds:
        xyz_sites = np.copy(dummy)
        Nsites += 1
        MTs_gen += 1
        print('MTs_gen :', MTs_gen)
        if MTs_gen == Nmts_to_gen: break

    for ilink in xyz_sites:
      if True: #abs(ilink[0]) < 5:
        axis_s = np.empty((s.size,3))
        normal = -ilink / np.linalg.norm(ilink)
        axis_s[:,0] = normal[0] * s
        axis_s[:,1] = normal[1] * s
        axis_s[:,2] = normal[2] * s
        axis_s = axis_s * prams.MT_length / 2 + ilink
        fib = fiber.fiber(num_points = options.num_points,
                          num_points_max = options.num_points_max,
                          num_points_finite_diff = options.num_points_finite_diff,
                          dt = options.dt,
                          E = prams.Efib,
                          length = prams.MT_length,
                          adaptive_num_points = options.adaptive_num_points,
                          viscosity = prams.eta)
        fib.ID = 'fibers'
        fib.x = axis_s
        fibers.append(fib)
  if prams.fiber_resume is not None:
    fibers_info, fibers_coor = read_fibers_file.read_fibers_file(prams.fiber_resume)
    # Create each fiber structure of type structure
    offset = 0
    for i in range(len(fibers_info)):
      if len(fibers_info[i]) > 0: num_points = fibers_info[i][0]
      if len(fibers_info[i]) > 2: length = fibers_info[i][2]
      
      fib_x = fibers_coor[offset : offset + num_points]
      fib_num_points = num_points

      fib = fiber.fiber(num_points = fib_num_points,
                        num_points_max = options.num_points_max,
                        num_points_finite_diff = options.num_points_finite_diff,
                        dt = options.dt,
                        E = prams.Efib,
                        length = length,
                        adaptive_num_points = options.adaptive_num_points,
                        viscosity = prams.eta)
      fib.ID = 'fibers'
      fib.x = fib_x
      offset += num_points
      # Append fiber to total bodies list
      fibers.append(fib)
  # Save additional info
  fibers_types.append(len(fibers))
  fibers_names.append('fibers')

  num_of_fibers_types = len(fibers_types)
  num_fibers = len(fibers)
  Nfibers_markers = sum([x.num_points for x in fibers])
  # Save bodies information
  with open(options.output_name + '_fibers.info', 'w') as f:
    f.write('num_of_fibers_types  ' + str(num_of_fibers_types) + '\n')
    f.write('fibers_names         ' + str(fibers_names) + '\n')
    f.write('fibers_types         ' + str(fibers_types) + '\n')
    f.write('num_fibers           ' + str(num_fibers) + '\n')
    f.write('num_fibers_markers   ' + str(Nfibers_markers) + '\n')


  # Open config files
  buffering = 100
  if len(fibers_types) > 0:
    f_fibers_ID = []
    for i, ID in enumerate(fibers_names):
      if options.output_txt_files:
        name = options.output_name + '_' + ID + '_fibers.txt'
      else:
        name = options.output_name + '_' + ID + '.fibers'

      f = open(name, 'wb', buffering=int(buffering))
      f_fibers_ID.append(f)

      if options.isaveForces:
        name = options.output_name + '_' + ID + '_fibers_repulsion_force.txt'
        f = open(name, 'wb', buffering=int(buffering))
        f_fibers_forces_ID.append(f)

        name = options.output_name + '_' + ID + '_fibers_motor_force.txt'
        f = open(name, 'wb', buffering=int(buffering))
        f_fibers_forces_ID.append(f)

  if len(bodies_types) > 0:
    f_bodies_ID = []
    for i, ID in enumerate(bodies_names):
      if options.output_txt_files:
        name = options.output_name + '_' + ID + '_clones.txt'
      else:
        name = options.output_name + '_' + ID + '.clones'

      f = open(name, 'wb', buffering=int(buffering))
      f_bodies_ID.append(f)

      if options.isaveForces:
        name = options.output_name + '_' + ID + '_clones_repulsion_force.txt'
        f = open(name, 'wb', buffering=int(buffering))
        f_bodies_forces_ID.append(f)

        name = options.output_name + '_' + ID + '_clones_links_force.txt'
        f = open(name, 'wb', buffering=int(buffering))
        f_bodies_forces_ID.append(f)

        name = options.output_name + '_' + ID + '_clones_links_torque.txt'
        f = open(name, 'wb', buffering=int(buffering))
        f_bodies_forces_ID.append(f)

  if None:
    f_molecular_motors_ID = []
    if options.output_txt_files:
      name = options.output_name + '_motors_attached_ends.txt'
    else:
      name = options.output_name + '.motors_attached_ends'
    f = open(name, 'wb', buffering=int(buffering))
    f_molecular_motors_ID.append(f)
    if options.output_txt_files:
      name = options.output_name + '_motors_base.txt'
    else:
      name = options.output_name + '.motors_base'
    f = open(name, 'wb', buffering=int(buffering))
    f_molecular_motors_ID.append(f)
    if options.output_txt_files:
      name = options.output_name + '_motors_head.txt'
    else:
      name = options.output_name + '.motors_head'
    f = open(name, 'wb', buffering=int(buffering))
    f_molecular_motors_ID.append(f)
    #if options.output_txt_files:
    #  name = options.output_name + '_molecular_motor_attached_ends.txt'
    #else:
    #  name = options.output_name + '.molecular_motor_attached_ends'
    #f = open(name, 'wb', buffering=int(buffering))
    #f_molecular_motors_ID.append(f)


  # Update bodies and fix links
  for k, b in enumerate(bodies):
    b.location = b.location_new
    b.orientation = b.orientation_new
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
      fib = fibers[offset_fiber]
      dx = fib.x_new[0] - link - b.location
      fib.x_new -= dx

  # Files for bodies and MMs on them
  if len(bodies_types_mm_attached) > 0:
    for i, ID in enumerate(bodies_names_mm_attached):
      if options.output_txt_files:
        name = options.output_name + '_' + ID + '_mm_clones.txt'
      else:
        name = options.output_name + '_' + ID + '.mm_clones'

      f = open(name, 'wb', buffering=int(buffering))
      f_bodies_mm_attached_ID.append(f)

      if options.isaveForces:
        name = options.output_name + '_' + ID + '_mm_clones_repulsion_force.txt'
        f = open(name, 'wb', buffering=int(buffering))
        f_bodies_mm_attached_forces_ID.append(f)

      if options.output_txt_files:
        name = options.output_name + '_' + ID + '_mm_moving_clone_base.txt'
      else:
        name = options.output_name + '_' + ID + '.mm_moving_clone_base'
      f = open(name, 'wb', buffering=int(buffering))
      f_mm_on_moving_surf_ID.append(f)

      if options.output_txt_files:
        name = options.output_name + '_' + ID + '_mm_moving_clone_head.txt'
      else:
        name = options.output_name + '_' + ID + '.mm_moving_clone_head'
      f = open(name, 'wb', buffering=int(buffering))
      f_mm_on_moving_surf_ID.append(f)

      if options.output_txt_files:
        name = options.output_name + '_' + ID + '_mm_moving_attached_ends.txt'
      else:
        name = options.output_name + '_' + ID + '.mm_moving_attached_ends'
      f = open(name, 'wb', buffering=int(buffering))
      f_mm_on_moving_surf_ID.append(f)



  # File to write time step size, # of time steps, number of files and bodies, and links
  name = options.output_name + '_time_system_size.txt'
  f_time_system = open(name, 'wb', buffering=int(buffering))

  return fibers, bodies, MM, fibers_names, bodies_names, fibers_types, bodies_types, f_fibers_ID, f_bodies_ID, f_molecular_motors_ID, f_time_system, f_fibers_forces_ID, f_bodies_forces_ID, MM_on_moving_surf, bodies_mm_attached, bodies_types_mm_attached, bodies_names_mm_attached, f_bodies_mm_attached_ID, f_bodies_mm_attached_forces_ID, f_mm_on_moving_surf_ID

def initialize_manually(fibers, bodies, molecular_motors, options, prams):

  # This assumes only one type of each object
  f_fibers_ID = []
  fibers_names = []
  fibers_types = []
  f_fibers_forces_ID = []
  buffering = 100
  if fibers is not None:
    if options.output_txt_files:
      name = options.output_name + '_fibers.txt'
    else:
      name = options.output_name + '.fibers'
    f = open(name, 'wb', buffering=int(buffering))
    f_fibers_ID.append(f)

    if options.isaveForces:
      name = options.output_name + '_' + ID + '_fibers_repulsion_force.txt'
      f = open(name, 'wb', buffering=int(buffering))
      f_fibers_forces_ID.append(f)

      name = options.output_name + '_' + ID + '_fibers_motor_force.txt'
      f = open(name, 'wb', buffering=int(buffering))
      f_fibers_forces_ID.append(f)

    fibers_names.append('one')
    fibers_types.append(len(fibers))
  else:
    fibers = []


  f_bodies_ID = []
  bodies_names = []
  bodies_types = []
  f_bodies_forces_ID = []
  if bodies is not None:
    if options.output_txt_files:
      name = options.output_name + '_bodies.txt'
    else:
      name = options.output_name + '.bodies'
    f = open(name, 'wb', buffering=int(buffering))
    f_bodies_ID.append(f)
    bodies_names.append('one')
    bodies_types.append(len(bodies))

    if options.isaveForces:
      name = options.output_name + '_' + ID + '_clones_repulsion_force.txt'
      f = open(name, 'wb', buffering=int(buffering))
      f_bodies_forces_ID.append(f)

      name = options.output_name + '_' + ID + '_clones_links_force.txt'
      f = open(name, 'wb', buffering=int(buffering))
      f_bodies_forces_ID.append(f)

      name = options.output_name + '_' + ID + '_clones_links_torque.txt'
      f = open(name, 'wb', buffering=int(buffering))
      f_bodies_forces_ID.append(f)
  else:
    bodies = []

  f_molecular_motors_ID = []
  if molecular_motors is not None:
    if options.output_txt_files:
      name = options.output_name + '_molecular_motors.txt'
    else:
      name = options.output_name + '.molecular_motors'
    f = open(name, 'wb', buffering=int(buffering))
    f_molecular_motors_ID.append(f)
    if options.output_txt_files:
      name = options.output_name + '_molecular_motors_base.txt'
    else:
      name = options.output_name + '.molecular_motors_base'
    f = open(name, 'wb', buffering=int(buffering))
    f_molecular_motors_ID.append(f)
    if options.output_txt_files:
      name = options.output_name + '_molecular_motors_head.txt'
    else:
      name = options.output_name + '.molecular_motors_head'
    f = open(name, 'wb', buffering=int(buffering))
    f_molecular_motors_ID.append(f)
  else:
    molecular_motors = []


   #  File to write time step size, # of time steps, number of files and bodies, and links
  name = options.output_name + '_time_system_size.txt'
  f_time_system = open(name, 'wb', buffering=int(buffering))

  return fibers, bodies, molecular_motors, fibers_names, bodies_names, fibers_types, bodies_types, f_fibers_ID, f_bodies_ID, f_molecular_motors_ID, f_time_system, f_fibers_forces_ID, f_bodies_forces_ID
