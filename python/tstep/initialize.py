
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
    iFiberOnly = False,
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
               fiber_body_attached = True,
               growing = None,
               motor_left = 0.8,
               motor_right = 0.8,
               only_body_radius = None,
               iEllipsoid = False):
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
    
def initialize_from_file(input_file,options,prams):
  '''
  Read data from input files, initialize fibers, bodies and molecular motors
  '''
  # Read input file
  read = read_input.ReadInput(input_file)
    
  # Copy input file to output
  subprocess.call(["cp", input_file, options.output_name + '.inputfile'])

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

  MM_on_moving_surf = []
  bodies_mm_attached = []
  bodies_types_mm_attached = []
  bodies_names_mm_attached = []
  f_bodies_mm_attached_ID = []
  f_bodies_mm_attached_forces_ID = []
  f_mm_on_moving_surf_ID = []

  for ID, structure in enumerate(read.structures_body_fibers):
    # Read fibers files
    fibers_info, fibers_coor = read_fibers_file.read_fibers_file(structure[2])

    # Create each fiber structure of type structure
    offset = 0
    for i in range(len(fibers_info)):
      if len(fibers_info[i]) > 0:
        num_points = fibers_info[i][0]
      else:
        num_points = read.num_points_fibers
      # E: bending modulus 
      if prams.Efib is not None:
         E = prams.Efib
      else:
         if len(fibers_info[i]) > 1:
          E = fibers_info[i][1]
         else:
          E = read.E
      if len(fibers_info[i]) > 2:
        length = fibers_info[i][2]
      else:
        length = read.length_fibers
      if prams.growing is not None:
        growing = prams.growing
      elif len(fibers_info[i]) > 3:
        growing = fibers_info[i][3]
      else:
        growing = np.random.randint(2)

      fib_x = fibers_coor[offset : offset + num_points]
      fib_num_points = num_points
      
      #alpha_in = np.flipud(cheb.cheb_parameter_space(num_points-1))
      if options.num_points != num_points and options.num_points is not None:
        alpha_in = np.linspace(-1,1,num_points)
        alpha_opt = np.linspace(-1,1,options.num_points)
        P_sample = bary.barycentricMatrix(alpha_in,alpha_opt)
        fib_x = np.dot(P_sample,fibers_coor[offset : offset + num_points])
        fib_num_points = options.num_points
      fib = fiber.fiber(num_points = fib_num_points,
                        num_points_max = options.num_points_max,
                        num_points_finite_diff = options.num_points_finite_diff,
                        dt = options.dt, 
                        E=E, 
                        length = length, 
                        epsilon = prams.epsilon, 
                        inonlocal = options.inonlocal,
                        ireparam = options.ireparam,
                        adaptive_num_points = options.adaptive_num_points,
                        tstep_order = options.order,
                        growing = growing,
                        viscosity = prams.eta)

      fib.ID = read.structures_body_fibers_ID[ID]
      fib.x = fib_x
      fib.v_growth = read.v_growth
      fib.v_shrink = read.v_shrink
      fib.rate_catastrophe = read.rate_catastrophe
      fib.rate_rescue = read.rate_rescue
      fib.rate_seed = read.rate_seed
      fib.rate_catastrophe_stall = read.rate_catastrophe_stall
      fib.force_stall = read.force_stall
      fib.length_min = read.length_min
      offset += num_points

      # Append fiber to total bodies list
      fibers.append(fib)
              
    # Save additional info
    fibers_types.append(i+1)
    fibers_names.append(read.structures_body_fibers_ID[ID])

    print('Creating structures body_fibers = ', structure[1])
    # Read vertex and clones files
    struct_ref_config = read_vertex_file.read_vertex_file(structure[0])
    num_bodies_struct, struct_locations, struct_orientations = read_clones_file.read_clones_file(structure[1])

    # Save bodies info
    bodies_types.append(num_bodies_struct)
    bodies_names.append(read.structures_body_fibers_ID[ID])


    # Create each body of type structure
    for i in range(num_bodies_struct):
      b = body.Body(struct_locations[i], struct_orientations[i], struct_ref_config, struct_ref_config, np.ones(struct_ref_config.size // 3))
      # b = body.Body(struct_locations[i], struct_orientations[i], struct_ref_config, 1.0)
      b.ID = read.structures_body_fibers_ID[ID]
      # Calculate body length for the RFD
      if b.Nblobs > 2000:
        b.body_length = 10.0
      elif i == 0:
        b.calc_body_length()
      else:
        b.body_length = bodies[-1].body_length
      bodies.append(b)

      # Read links file
      if len(structure) > 3:
        spring_constants, spring_constants_angle, links_location, axes = read_links_file.read_links_file(structure[3])
        bodies[-1].links_spring_constant = spring_constants
        bodies[-1].links_spring_constants_angle = spring_constants_angle
        bodies[-1].links_location = links_location
        bodies[-1].links_axis = axes
        bodies[-1].links_first_fibers = sum(fibers_types[:-1])
        bodies[-1].links_last_fibers = sum(fibers_types) 
        lin = links_location[0] 
        bodies[-1].radius = np.sqrt(lin[0]**2 + lin[1]**2 + lin[2]**2)


        # Calculate interfilament distance
        if links_location.size//3 > 1:
          dx = links_location[:,0] - links_location[:,0,None]
          dy = links_location[:,1] - links_location[:,1,None]
          dz = links_location[:,2] - links_location[:,2,None]
          dr = np.sqrt(dx**2 + dy**2 + dz**2)
          bodies[-1].dfilament = min(dr[0,1:])
          print('Interfilament distance = ', bodies[-1].dfilament)
        if options.body_quadrature_radius is None:
          bodies[-1].quadrature_radius = 4.0* bodies[-1].radius / 5.0 
        else:
          bodies[-1].quadrature_radius = options.body_quadrature_radius
      

      else:
        if prams.only_body_radius is not None:
          bodies[-1].radius = prams.only_body_radius
          bodies[-1].quadrature_radius = prams.only_body_radius
        else:
          bodies[-1].quadrature_radius = options.body_quadrature_radius
          bodies[-1].radius = options.body_quadrature_radius

      print('Body radius = ', bodies[-1].radius)
      print('Quadrature body radius = ', bodies[-1].quadrature_radius)

      
 
  for ID, structure in enumerate(read.structures):
    print('Creating structures = ', structure[1])

    # Read vertex and clones files
    struct_ref_config = read_vertex_file.read_vertex_file(structure[0])
    num_bodies_struct, struct_locations, struct_orientations = read_clones_file.read_clones_file(structure[1])

    # Save bodies info
    bodies_types.append(num_bodies_struct)
    bodies_names.append(read.structures_ID[ID])

    # Create each body of type structure
    for i in range(num_bodies_struct):
      b = body.Body(struct_locations[i], struct_orientations[i], struct_ref_config, struct_ref_config, np.ones(struct_ref_config.size // 3))
      #b = body.Body(struct_locations[i], struct_orientations[i], struct_ref_config, 1.0)
      b.ID = read.structures_ID[ID]
      # Calculate body length for the RFD
      if b.Nblobs > 2000:
        b.body_length = 10.0
      elif i == 0:
        b.calc_body_length()
      else:
        b.body_length = bodies[-1].body_length
      bodies.append(b)
      if prams.only_body_radius is not None:
        bodies[-1].radius = prams.only_body_radius
        bodies[-1].quadrature_radius = prams.only_body_radius
      else:
        bodies[-1].quadrature_radius = options.body_quadrature_radius
        bodies[-1].radius = options.body_quadrature_radius
      

  # Set some more variables
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

  # Create molecular motors

  if read.molecular_motor_config is not None:
    print('read.molecular_motor_config = ', read.molecular_motor_config)
    r_MM = np.loadtxt(read.molecular_motor_config)
    radius = read.molecular_motor_radius
    speed_0 = read.molecular_motor_speed
    force_stall = read.molecular_motor_force_stall
    spring_constant = read.molecular_motor_spring_constant
    rest_length = read.molecular_motor_rest_length
    bind_frequency = read.molecular_motor_bind_frequency
    unbind_frequency_0 = read.molecular_motor_unbind_frequency
    kernel_sigma = read.molecular_motor_kernel_sigma
    MM = molecular_motor.molecular_motor(r_MM, radius, speed_0, force_stall, spring_constant, rest_length, bind_frequency, unbind_frequency_0, kernel_sigma)
    name = options.output_name + '.force_generator.txt'
    if read.molecular_motor_attached_ends is not None:
      attached_ends = np.transpose(np.loadtxt(read.molecular_motor_attached_ends, dtype = float)).astype(int)
    else:
      attached_ends = np.zeros((MM.N, 2))
      attached_ends[:,0] = -1
      attached_ends[:,1] = -2
    
    if read.molecular_motor_s_head is not None:
      MM.s_head = np.loadtxt(read.molecular_motor_s_head)
    if read.molecular_motor_s_head is not None:
      MM.s_base = np.loadtxt(read.molecular_motor_s_base)

    if read.molecular_motor_head_config is not None:
      r_head_MM = np.loadtxt(read.molecular_motor_head_config)
      MM.x_head = r_head_MM.reshape((r_head_MM.size // 3, 3))

    MM.attached_base[:] = attached_ends[:,0]
    MM.attached_head[:] = attached_ends[:,1]

    with open(name, 'w') as f:
      if options.output_txt_files:
        mm_info = np.array([MM.x.size/3, radius, 0])
        np.savetxt(f,mm_info[None,:])
      else:
        f.write(str(MM.x.size/3) + '\n')
      np.savetxt(f,MM.x)

  # LOAD bodies on which MMs initialized, both moving
  for ID, structure in enumerate(read.mm_surface_structure):
    # structure[0]: clone
    # structure[1]: mm_base.dat file (forcegenerator.dat)
    # structure[2]: mm_head
    # structure[3]: attached ends
    # structure[4]: s_base
    # structure[5]: s_head
    print('Creating structures mm_surface_structure = ', structure[0])
    num_bodies_struct, struct_locations, struct_orientations = read_clones_file.read_clones_file(structure[0])

    # Save bodies info
    bodies_types_mm_attached.append(num_bodies_struct)
    bodies_names_mm_attached.append(read.mm_surface_structure_ID[ID])

    # Create each body of type structure
    for i in range(num_bodies_struct):
      b = body.Body(struct_locations[i], struct_orientations[i], struct_ref_config, struct_ref_config, np.ones(struct_ref_config.size // 3))
      # b = body.Body(struct_locations[i], struct_orientations[i], struct_ref_config, 1.0)
      b.ID = read.mm_surface_structure_ID[ID]
      # Calculate body length for the RFD
      if b.Nblobs > 2000:
        b.body_length = 10.0
      elif i == 0:
        b.calc_body_length()
      else:
        b.body_length = bodies_mm_attached[-1].body_length
      bodies_mm_attached.append(b)

    
    print('Creating structures mm_surface_structure = ', structure[1])
    r_MM = np.loadtxt(structure[1]) # mm_base
    if structure[2] == structure[1]:
      r_head_MM = np.copy(r_MM).reshape((r_MM.size // 3, 3))
    else:
      r_head_MM = np.loadtxt(structure[2]) # mm_head

    radius = read.molecular_motor_radius
    speed_0 = read.molecular_motor_speed
    force_stall = read.molecular_motor_force_stall
    spring_constant = read.molecular_motor_spring_constant
    rest_length = read.molecular_motor_rest_length
    bind_frequency = read.molecular_motor_bind_frequency
    unbind_frequency_0 = read.molecular_motor_unbind_frequency
    kernel_sigma = read.molecular_motor_kernel_sigma
    mm_moving = molecular_motor.molecular_motor(r_MM, radius, speed_0, force_stall, spring_constant, rest_length, bind_frequency, unbind_frequency_0, kernel_sigma)
    mm_moving.x_head = r_head_MM.reshape((r_head_MM.size // 3, 3))
    
    if len(structure) > 3:
      attached_ends = np.transpose(np.loadtxt(structure[3], dtype = float)).astype(int)
    else:
      attached_ends = np.zeros((mm_moving.N, 2))
      attached_ends[:,0] = -1
      attached_ends[:,1] = -2

    if len(structure) > 4:
      mm_moving.s_base = np.loadtxt(structure[4], dtype = float)
      mm_moving.s_head = np.loadtxt(structure[5], dtype = float)


    mm_moving.attached_base[:] = attached_ends[:,0]
    mm_moving.attached_head[:] = attached_ends[:,1]
    MM_on_moving_surf.append(mm_moving)

    # Determine body radius based on MM config (little smaller than where MMs are)
    mm_cent = np.mean(mm_moving.x,axis=0)
    bodies_mm_attached[-1].radius = np.sqrt((mm_moving.x[0,0]-mm_cent[0])**2 + (mm_moving.x[0,1]-mm_cent[1])**2 + (mm_moving.x[0,2]-mm_cent[2])**2)*0.833

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

  if read.molecular_motor_config is not None:
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
