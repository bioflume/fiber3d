
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
    from quaternion import quaternion
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
    dt_min = 1e-5,
    dt_max = 0.01,
    tol_tstep = 1e-2,
    tol_gmres = 1e-12,
    fiber_ds = 0.3/32,
    min_body_ds = 0.10,
    num_points = 32,
    num_points_finite_diff = 4,
    num_points_max = 96,
    random_seed = None,
    uprate =4,
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
    release_condition = 30000.0,
    iExternalForce = False,
    iCytoPulling = False,
    iNoSliding = False,
    dynInstability = True,
    iPeripheralNucleation = False,
    iComputeVelocity = False,
    ncompute_vel = 500,
    save_file = None):
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
    self.fiber_ds = fiber_ds
    self.num_points = num_points
    self.inonlocal = inonlocal
    self.output_name = output_name
    self.save_file = save_file
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
    self.iNoSliding = iNoSliding
    self.min_body_ds = min_body_ds
    self.dynInstability = dynInstability
    self.iPeripheralNucleation = iPeripheralNucleation
    self.iComputeVelocity = iComputeVelocity
    self.ncompute_vel = ncompute_vel
    self.uprate = uprate

class set_parameters(object):
  def __init__(self, eta = 1.0,
               epsilon = 1e-03,
               final_time = 1.0,
               Efib = None,
               time_max = None,
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
               fiber_body_attached = True,
               growing = 1,
               motor_left = 0.8,
               motor_right = 0.8,
               only_body_radius = None,
               viscosity = 1,
               nucleation_rate = 62.5,
               rate_catastrophe = 0.015,
               v_growth = 0.75,
               minL = 0.5,
               force_stall = 4.4,
               max_length = None,
               max_nuc_sites = None,
               attached_to_cortex = False,
               active_sites_idcs_file = None,
               passive_sites_idcs_file = None,
               occupied_sites_idcs_file = None,
               fib_sites_lengths_file = None,
               scale_life_time = 0.5,
               scale_vg = 0.75,
               resume_from_step = None,
               site_idcs_nucleating_file = None,
               site_idcs_dying_file = None,
               site_idcs_hinged_file = None,
               iRelaxationRun = False,
               when_relax = None,
               cent_location = None,
               body_viscosity_scale = None,
               Nmts = None,
               mt_length = None,
               body_radius = None,
               cyto_force = None):
    # FLOW PARAMETERS
    self.viscosity = viscosity # new fluid viscosity parameter
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
    self.time_max = time_max
    self.growing = growing
    self.motor_left = motor_left
    self.motor_right = motor_right
    self.only_body_radius = only_body_radius
    self.nucleation_rate = nucleation_rate
    self.rate_catastrophe = rate_catastrophe
    self.v_growth = v_growth
    self.minL = minL
    self.force_stall = force_stall
    self.max_length = max_length
    self.max_nuc_sites = max_nuc_sites
    self.resume_from_step = resume_from_step
    self.attached_to_cortex = attached_to_cortex
    self.active_sites_idcs_file = active_sites_idcs_file
    self.passive_sites_idcs_file = passive_sites_idcs_file
    self.fib_sites_lengths_file = fib_sites_lengths_file
    self.scale_life_time = scale_life_time
    self.scale_vg = scale_vg
    self.occupied_sites_idcs_file = occupied_sites_idcs_file
    self.site_idcs_nucleating_file = site_idcs_nucleating_file
    self.site_idcs_dying_file = site_idcs_dying_file
    self.site_idcs_hinged_file = site_idcs_hinged_file
    self.iRelaxationRun = iRelaxationRun
    self.when_relax = when_relax
    self.cent_location = cent_location
    self.body_viscosity_scale = body_viscosity_scale
    self.Nmts = Nmts
    self.mt_length = mt_length
    self.body_radius = body_radius
    self.cyto_force = cyto_force

def initialize_from_file(input_file,options,prams):
  '''
  Read data from input files, initialize fibers, bodies and molecular motors
  '''
  

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
  f_body_vels_ID = []
  f_bodies_forces_ID = []
  f_molecular_motors_ID = []

  MM_on_moving_surf = []
  bodies_mm_attached = []
  bodies_types_mm_attached = []
  bodies_names_mm_attached = []
  f_bodies_mm_attached_ID = []
  f_bodies_mm_attached_forces_ID = []
  f_mm_on_moving_surf_ID = []

  total_nuc_sites = 0

  struct_ref_config = np.array([0, 0, 0])
  orientation = [1, 0, 0, 0]
  norm_orientation = np.linalg.norm(orientation)
  struct_orientation = quaternion.Quaternion(orientation / norm_orientation)

  b = body.Body(np.array([0,0,0]), struct_orientation, struct_ref_config, struct_ref_config, np.ones(struct_ref_config.size // 3))
  b.ID = 'body'
  b.radius = prams.body_radius
  b.quadrature_radius = options.body_quadrature_radius
  bodies.append(b)
  bodies_types.append(1)
  bodies_names.append('body')

  Nmts_to_gen = prams.Nmts
  MTs_gen, ntrials = 0, 0
  linkr = bodies[0].radius
  min_ds, Nsites = 0.1, 0
  s = np.linspace(0, 2, options.num_points)
  while MTs_gen < Nmts_to_gen and ntrials <= 100:
    # Sample a point
    xq2 = linkr**2 * np.random.randn()
    yq2 = linkr**2 * np.random.randn()
    zq2 = linkr**2 * np.random.randn()

    # Project it onto the surface
    d = np.sqrt(xq2**2 / linkr**2 + yq2**2 / linkr**2 + zq2**2 / linkr**2)
    xq = xq2 / d
    yq = yq2 / d
    zq = zq2 / d
    ilink = np.array([xq, yq, zq])

    iPlace = True
    
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
      axis_s = axis_s * prams.mt_length / 2 + ilink + bodies[0].location

      fib = fiber.fiber(num_points = options.num_points,
                        num_points_max = options.num_points_max,
                        num_points_finite_diff = options.num_points_finite_diff,
                        dt = options.dt,
                        E = prams.Efib,
                        length = prams.mt_length,
                        adaptive_num_points = options.adaptive_num_points,
                        viscosity = prams.eta)
      fib.attached_to_body = 0
      fib.nuc_site_idx = int(Nsites-1)
      fib.ID = 'fibers'
      fib.x = axis_s
      fibers.append(fib)        
  
  bodies[0].nuc_sites = xyz_sites
  bodies[0].active_sites_idcs = np.arange(xyz_sites.size//3).tolist()
  bodies[0].passive_sites_idcs = []
  bodies[0].min_ds = min_ds
  total_nuc_sites += xyz_sites.size//3
  print('Total nuc sites: ', total_nuc_sites)

  
  # Set some more variables
  fibers_names.append('fibers')
  fibers_types.append(len(fibers))
  num_of_fibers_types = len(fibers_types)
  num_fibers = len(fibers)
  Nfibers_markers = sum([x.num_points for x in fibers])

  # Set random generator state
  if options.random_seed is not None:
    if isinstance(options.random_seed, int):
      np.random.seed(int(options.random_seed))
    else: # then it is a file
      print(options.random_seed)
      with open(options.random_seed, 'rb') as f:
        np.random.set_state(cpickle.load(f))
  else:
    np.random.seed(None)
  # Save random generator state
  with open(options.output_name + 'step_0'+ '.random_state', 'wb') as f:
    cpickle.dump(np.random.get_state(), f)

  # Save bodies information
  with open(options.output_name + '_fibers.info', 'w') as f:
    f.write('number_of_bodies     ' + str(len(bodies)))
    f.write('num_of_nuc_sites     ' + str(total_nuc_sites) + '\n')
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
    f_body_vels_ID = []
    for i, ID in enumerate(bodies_names):
      if options.output_txt_files:
        name = options.output_name + '_' + ID + '_clones.txt'
      else:
        name = options.output_name + '_' + ID + '.clones'

      f = open(name, 'wb', buffering=int(buffering))
      f_bodies_ID.append(f)

      name = options.output_name + '_' + ID + 'velocity_clones.txt'
      f = open(name, 'wb', buffering=int(buffering))
      f_body_vels_ID.append(f)

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

  return fibers, bodies, MM, fibers_names, bodies_names, fibers_types, bodies_types, f_fibers_ID, f_bodies_ID, f_molecular_motors_ID, f_time_system, f_fibers_forces_ID, f_bodies_forces_ID, MM_on_moving_surf, bodies_mm_attached, bodies_types_mm_attached, bodies_names_mm_attached, f_bodies_mm_attached_ID, f_bodies_mm_attached_forces_ID, f_mm_on_moving_surf_ID, f_body_vels_ID

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
