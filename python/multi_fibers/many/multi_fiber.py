from __future__ import print_function
import numpy as np
import sys
import argparse
import subprocess
import time
import matplotlib.pyplot as plt
try:
  import cPickle as cpickle
except:
  try:
    import cpickle
  except:
    import _pickle as cpickle
try:
  from mpi4py import MPI
except ImportError:
  print('It didn\'t find mpi4py')
try:
  from numba import njit, prange
  numba_found = True
except ImportError:
  print('Numba not found')

  
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
    from fiber import fiber
    from integrators import integrators
    from kernels import kernels
    from body import body
    # from force_generator import force_generator as fg
    from molecular_motor import molecular_motor
    #from lib import periodic_fmm as fmm
    found_functions = True
  except ImportError:
    path_to_append += '../'
    print('searching functions in path ', path_to_append)
    sys.path.append(path_to_append)
    if len(path_to_append) > 21:
      print('\nProjected functions not found. Edit path in multi_bodies.py')
      sys.exit()





if __name__ == '__main__':
  # Get command line arguments
  parser = argparse.ArgumentParser(description='Run a multi-fiber simulation and save trajectory.')
  parser.add_argument('--input-file', dest='input_file', type=str, default='run5002.0.0.inputfile', help='name of the input file')
  parser.add_argument('--verbose', action='store_true', help='print convergence information')
  args=parser.parse_args()
  input_file = args.input_file
  
  # Read input file
  read = read_input.ReadInput(input_file)
  
  # Copy input file to output
  subprocess.call(["cp", input_file, read.output_name + '.inputfile'])

  # Set random generator state
  if read.random_state is not None:
    with open(read.random_state, 'rb') as f:
      np.random.set_state(cPickle.load(f))
  elif read.seed is not None:
    np.random.seed(int(read.seed))
    if numba_found:
      seed_numba = np.random.randint(np.iinfo(np.uint32).max)
      @njit(parallel=False)
      def init_random_numba(seed):
        np.random.seed(seed)
      init_random_numba(seed_numba)
  
  # Save random generator state
  with open(read.output_name + '.random_state', 'wb') as f:
    cpickle.dump(np.random.get_state(), f)

  # Create fibers
  fibers = []
  fibers_types = []
  fibers_names = []
  for ID, structure in enumerate(read.structures_fibers):
    print('Creating structures fibers = ', structure)
    # Read fibers files
    fibers_info, fibers_coor = read_fibers_file.read_fibers_file(structure)

    # Create each fiber structure of type structure
    offset = 0
    for i in range(len(fibers_info)):
      if len(fibers_info[i]) > 0:
        num_points = fibers_info[i][0]
      else:
        num_points = read.num_points_fibers
      if len(fibers_info[i]) > 1:
        E = fibers_info[i][1]
      else:
        E = read.E
      if len(fibers_info[i]) > 2:
        length = fibers_info[i][2]
      else:
        length = read.length_fibers
      fib = fiber.fiber(num_points = num_points, 
                        dt = read.dt, 
                        E=E, 
                        length = length, 
                        epsilon = read.epsilon, 
                        num_points_finite_diff = read.num_points_finite_diff)
      fib.ID = read.structures_fibers_ID[ID]
      fib.x = fibers_coor[offset : offset + num_points]
      fib.set_BC(BC_start_0='force', BC_start_vec_0=np.array([0.0, 0.0, 0.0]))
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
    fibers_names.append(read.structures_fibers_ID[ID])

      
  # Create body_fibers
  bodies = []
  body_types = []
  body_names = []
  for ID, structure in enumerate(read.structures_body_fibers):
    print('Creating structures body_fibers = ', structure[1])
    # Read fibers files
    fibers_info, fibers_coor = read_fibers_file.read_fibers_file(structure[2])

    # Create each fiber structure of type structure
    offset = 0
    for i in range(len(fibers_info)):
      if len(fibers_info[i]) > 0:
        num_points = fibers_info[i][0]
      else:
        num_points = read.num_points_fibers
      if len(fibers_info[i]) > 1:
        E = fibers_info[i][1]
      else:
        E = read.E
      if len(fibers_info[i]) > 2:
        length = fibers_info[i][2]
      else:
        length = read.length_fibers
      fib = fiber.fiber(num_points = num_points, 
                        dt = read.dt, 
                        E=E, 
                        length = length, 
                        epsilon = read.epsilon, 
                        num_points_finite_diff = read.num_points_finite_diff)
      fib.ID = read.structures_body_fibers_ID[ID]
      fib.x = fibers_coor[offset : offset + num_points]
      fib.set_BC(BC_start_0='velocity', BC_start_1='angular_velocity')
      # fib.set_BC(BC_start_0='velocity')
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

    # Read vertex and clones files
    struct_ref_config = read_vertex_file.read_vertex_file(structure[0])
    num_bodies_struct, struct_locations, struct_orientations = read_clones_file.read_clones_file(structure[1])

    # Save bodies info
    body_types.append(num_bodies_struct)
    body_names.append(read.structures_body_fibers_ID[ID])


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
    spring_constants, spring_constants_angle, links_location, axes = read_links_file.read_links_file(structure[3])
    bodies[-1].links_spring_constant = spring_constants
    bodies[-1].links_spring_constants_angle = spring_constants_angle
    bodies[-1].links_location = links_location
    bodies[-1].links_axis = axes
    bodies[-1].links_first_fibers = sum(fibers_types[:-1])
    bodies[-1].links_last_fibers = sum(fibers_types) 


  for ID, structure in enumerate(read.structures):
    print('Creating structures = ', structure[1])
 
    # Read vertex and clones files
    struct_ref_config = read_vertex_file.read_vertex_file(structure[0])
    num_bodies_struct, struct_locations, struct_orientations = read_clones_file.read_clones_file(structure[1])

    # Save bodies info
    body_types.append(num_bodies_struct)
    body_names.append(read.structures_ID[ID])

    # Create each body of type structure
    for i in range(num_bodies_struct):
      b = body.Body(struct_locations[i], struct_orientations[i], struct_ref_config, 1.0)
      b.ID = read.structures_ID[ID]
      # Calculate body length for the RFD
      if b.Nblobs > 2000:
        b.body_length = 10.0
      elif i == 0:
        b.calc_body_length()
      else:
        b.body_length = bodies[-1].body_length
      bodies.append(b)

  # Set some more variables
  num_of_fibers_types = len(fibers_types)
  num_fibers = len(fibers)
  Nfibers_markers = sum([x.num_points for x in fibers])

  # Save bodies information
  with open(read.output_name + '.fibers_info', 'w') as f:
    f.write('num_of_fibers_types  ' + str(num_of_fibers_types) + '\n')
    f.write('fibers_names         ' + str(fibers_names) + '\n')
    f.write('fibers_types         ' + str(fibers_types) + '\n')
    f.write('num_fibers           ' + str(num_fibers) + '\n')
    f.write('num_fibers_markers   ' + str(Nfibers_markers) + '\n')

  # Create integrator
  integrator = integrators.integrator(read.scheme, 
                                      fibers, 
                                      read.solver_tolerance, 
                                      Nfibers_markers, 
                                      verbose = args.verbose, 
                                      p_fmm = read.p_fmm, 
                                      eta = read.eta,
                                      bodies = bodies)

  # Create molecular motors
  if read.molecular_motor_config is not None:
    print('read.molecular_motor_config = ', read.molecular_motor_config)
    r_MM = np.loadtxt(read.molecular_motor_config, skiprows=1)
    radius = read.molecular_motor_radius
    speed_0 = read.molecular_motor_speed
    force_stall = read.molecular_motor_force_stall
    spring_constant = read.molecular_motor_spring_constant
    rest_length = read.molecular_motor_rest_length
    bind_frequency = read.molecular_motor_bind_frequency
    unbind_frequency_0 = read.molecular_motor_unbind_frequency
    kernel_sigma = read.molecular_motor_kernel_sigma
    MM = molecular_motor.molecular_motor(r_MM, radius, speed_0, force_stall, spring_constant, rest_length, bind_frequency, unbind_frequency_0, kernel_sigma)
    name = read.output_name + '.force_generator'
    MM.attached_base[:] = -1
    integrator.molecular_motor = MM
    with open(name, 'w') as f:
      f.write(str(MM.x.size / 3) + '\n')
      np.savetxt(f, MM.x)
    
  # Open config files
  if len(fibers_types) > 0:
    buffering = max(1, min(fibers_types) * read.n_steps // read.n_save // 200)
    f_fibers_ID = []
    for i, ID in enumerate(fibers_names):
      name = read.output_name + '.' + ID + '.fibers'
      f = open(name, 'wb', buffering=buffering)
      f_fibers_ID.append(f)
    
  if len(body_types) > 0:   
    buffering = max(1, min(body_types) * read.n_steps // read.n_save // 200)
    f_bodies_ID = []
    for i, ID in enumerate(body_names):
      name = read.output_name + '.' + ID + '.clones'
      f = open(name, 'wb', buffering=buffering)
      f_bodies_ID.append(f)

  if read.molecular_motor_config is not None:
    buffering = max(1, r_MM.size * read.n_steps // read.n_save // 200)
    f_molecular_motors_ID = []
    name = read.output_name + '.molecular_motors'
    f = open(name, 'wb', buffering=buffering)
    f_molecular_motors_ID.append(f)
    name = read.output_name + '.molecular_motors_base'
    f = open(name, 'wb', buffering=buffering)
    f_molecular_motors_ID.append(f)
    name = read.output_name + '.molecular_motors_head'
    f = open(name, 'wb', buffering=buffering)
    f_molecular_motors_ID.append(f)

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



  # Loop over time steps
  start_time = time.time()  
  current_time = 0.0
  final_time = read.dt * read.n_steps
  time_save = 0.0
  step = 0
  steps_rejected = 0
  dt = read.dt 
  timer.timer('zzz_loop')
  while current_time < final_time:
    # Save data if...
    if current_time >= time_save:
      time_save += read.n_save * read.dt
      elapsed_time = time.time() - start_time
      print('Integrator = ', read.scheme, ', step = ', step, ', time = ', current_time, ', dt = ', dt, ', wallclock time = ', time.time() - start_time)
      # For each type of structure save locations and orientations to one file
      if True:
        # Save bodies
        bodies_offset = 0
        for i, ID in enumerate(body_names):
          f_bodies_ID[i].write((str(body_types[i]) + '\n').encode('utf-8'))
          for j in range(body_types[i]):
            orientation = bodies[bodies_offset + j].orientation.entries
            f_bodies_ID[i].write(('%s %s %s %s %s %s %s \n' % (bodies[bodies_offset + j].location[0],
                                                              bodies[bodies_offset + j].location[1],
                                                              bodies[bodies_offset + j].location[2],
                                                              orientation[0],
                                                              orientation[1],
                                                              orientation[2],
                                                              orientation[3])).encode('utf-8'))
          bodies_offset += body_types[i]

        # Save fibers
        fiber_offset = 0
        for i, ID in enumerate(fibers_names):
          f_fibers_ID[i].write((str(fibers_types[i]) + '\n').encode('utf-8'))
          for j in range(fibers_types[i]):
            f_fibers_ID[i].write(('%s %s %s\n' % (fibers[fiber_offset + j].num_points, fibers[fiber_offset + j].E, fibers[fiber_offset + j].length)).encode('utf-8'))
            np.savetxt(f_fibers_ID[i], np.concatenate((fibers[fiber_offset + j].x, fibers[fiber_offset + j].tension[:,None]), axis=1)) 
          fiber_offset += fibers_types[i]

        # Save molecular motors
        if read.molecular_motor_config is not None:
          f_molecular_motors_ID[0].write((str(MM.N) + '\n').encode('utf-8'))
          np.savetxt(f_molecular_motors_ID[0], MM.x)
          f_molecular_motors_ID[1].write((str(MM.N) + '\n').encode('utf-8'))
          np.savetxt(f_molecular_motors_ID[1], MM.x_base)
          f_molecular_motors_ID[2].write((str(MM.N) + '\n').encode('utf-8'))
          np.savetxt(f_molecular_motors_ID[2], MM.x_head)

    # Save old configuration
    for fib in fibers:
      fib.x_old = np.copy(fib.x)
    for b in bodies:
      b.location_old = np.copy(b.location)
      b.orientation_old = np.copy(b.orientation)

    # Updating the system
    integrator.advance_time_step(dt)
  
    # Check stretching
    timer.timer('check_error')
    fiber_error = np.zeros(len(fibers))
    fiber_error_max = 0
    fiber_error_max_index = -1
    for fib_k, fib in enumerate(fibers):
      fib.xs = np.dot(fib.D_1, fib.x_new)
      fiber_error[fib_k] = abs(max(np.sqrt(fib.xs[1:-1,0]**2 + fib.xs[1:-1,1]**2 + fib.xs[1:-1,2]**2) - 1.0, key=abs))
      if fiber_error[fib_k] > fiber_error_max:
        fiber_error_max = fiber_error[fib_k]
        fiber_error_max_index = fib_k
    print('stretching_error = ', fiber_error_max, ', fiber_error = ', fiber_error_max_index, ', dt = ', dt)
    timer.timer('check_error')

    dt_old = dt
    max_acceptable_error = 1e-02
    if fiber_error_max <= max_acceptable_error:
      timer.timer('update')

      # Update fibers
      for fib_k, fib in enumerate(fibers):
        if fiber_error[fib_k] >= max_acceptable_error * 10:
          print('fiber = ', fib_k, ', error before correction = ', fiber_error[fib_k])
          fib.correct()
          xs = np.dot(fib.D_1, fib.x)
          print('fiber = ', fib_k, ', error after correction = ', abs(max(np.sqrt(xs[6:-1,0]**2 + xs[6:-1,1]**2 + xs[6:-1,2]**2) - 1.0, key=abs)))
          print('fiber = ', fib_k, ', error after correction = ', abs(max(np.sqrt(xs[1:-1,0]**2 + xs[1:-1,1]**2 + xs[1:-1,2]**2) - 1.0, key=abs)))
        else:
          fib.x = np.copy(fib.x_new)
      current_time += dt
      timer.timer('update')
      timer.timer('set_dt')
      if fiber_error_max <= max_acceptable_error * 0.01:
        dt = min(read.dt, dt * 1.1)
      elif fiber_error_max > max_acceptable_error * 0.9:
        dt = min(max(dt / 2.0, 0.01 * read.dt), dt * 1.01)
      else:
        dt = min(max(dt / 2.0, 0.01 * read.dt + (1.0 - fiber_error_max / max_acceptable_error)**1.5 * (0.99 * read.dt)), dt * 1.05)

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
          dx = fib.x[0] - link - b.location
          fib.x -= dx
      timer.timer('set_dt')
    else:
      print('Step rejected!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! dt = ', dt, '\n')
      # dt = min(0.01 * read.dt, dt / 2.0)
      dt = dt / 2.0
      steps_rejected += 1
      for fib in fibers:
        fib.length = fib.length_previous
        fib.D_1 = fib.D_1_0 * (2.0 / fib.length)
        fib.D_2 = fib.D_2_0 * (2.0 / fib.length)**2
        fib.D_3 = fib.D_3_0 * (2.0 / fib.length)**3
        fib.D_4 = fib.D_4_0 * (2.0 / fib.length)**4
        fib.s = (1.0 + fib.alpha) * (fib.length / 2.0)
        fib.weights = fib.weights_0 * (fib.length / 2.0)
      fib = fibers[fiber_error_max_index]
      print(np.sqrt(fib.xs[:,0]**2 + fib.xs[:,1]**2 + fib.xs[:,2]**2) - 1.0)
      print(np.dot(fib.D_2[0,:], fib.x_new))
      print(np.dot(fib.D_2[-1,:], fib.x_new))
      if steps_rejected > 1 * read.n_steps or dt < read.dt / 1e+06:
        fib = fibers[fiber_error_max_index]
        # print(np.sqrt(fib.xs[:,0]**2 + fib.xs[:,1]**2 + fib.xs[:,2]**2) - 1.0)
        break

    for fib in fibers:
      fib.beta = fib.beta_0 / dt
        
    step += 1      
  timer.timer('zzz_loop')



  if current_time >= time_save:
    time_save += read.n_save * read.dt
    elapsed_time = time.time() - start_time
    print('Integrator = ', read.scheme, ', step = ', step, ', time = ', current_time, ', dt = ', dt, ', wallclock time = ', time.time() - start_time)
    # For each type of structure save locations and orientations to one file
    if True:
      # Save bodies
      bodies_offset = 0
      for i, ID in enumerate(body_names):
        f_bodies_ID[i].write((str(body_types[i]) + '\n').encode('utf-8'))
        for j in range(body_types[i]):
          orientation = bodies[bodies_offset + j].orientation.entries
          f_bodies_ID[i].write(('%s %s %s %s %s %s %s \n' % (bodies[bodies_offset + j].location[0],
                                                            bodies[bodies_offset + j].location[1],
                                                            bodies[bodies_offset + j].location[2],
                                                            orientation[0],
                                                            orientation[1],
                                                            orientation[2],
                                                            orientation[3])).encode('utf-8'))
        bodies_offset += body_types[i]

      # Save fibers
      fiber_offset = 0
      for i, ID in enumerate(fibers_names):
        f_fibers_ID[i].write((str(fibers_types[i]) + '\n').encode('utf-8'))
        for j in range(fibers_types[i]):
          f_fibers_ID[i].write(('%s %s %s\n' % (fibers[fiber_offset + j].num_points, fibers[fiber_offset + j].E, fibers[fiber_offset + j].length)).encode('utf-8'))
          np.savetxt(f_fibers_ID[i], np.concatenate((fibers[fiber_offset + j].x, fibers[fiber_offset + j].tension[:,None]), axis=1)) 
        fiber_offset += fibers_types[i]


  # Close config files
  if len(fibers_types) > 0:
    for f_ID in f_fibers_ID:
      f_ID.close()
  if len(body_types) > 0:
    for f_ID in f_bodies_ID:
      f_ID.close()
  if read.molecular_motor_config is not None:
    for f_ID in f_molecular_motors_ID:
      f_ID.close()

  print('steps rejected = ', steps_rejected)

  # Save wallclock time 
  with open(read.output_name + '.time', 'w') as f:
    f.write(str(time.time() - start_time) + '\n')
  timer.timer(' ', print_all = True, output_file=read.output_name+'.timers')
  print('\n\n\n# End')


