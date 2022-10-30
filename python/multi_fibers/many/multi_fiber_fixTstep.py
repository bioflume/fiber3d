from __future__ import print_function
import numpy as np
import sys
import argparse
import subprocess
import time
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

try:
  import cPickle as cpickle
except:
  try:
    import cpickle
  except:
    import _pickle as cpickle

  
# Find project functions
found_functions = False
sys.path.append('../../')

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
 

if __name__ == '__main__':
  # Simulation information
  input_file = 'run2fibers.inputfile'
  # Whether output details in integrator
  flagVerbose = False
  
  
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
  
  # Save random generator state
  with open(read.output_name + '.random_state', 'wb') as f:
    cpickle.dump(np.random.get_state(), f)


  # Create body_fibers
  bodies = []
  body_types = []
  body_names = []
  fibers = []
  fibers_types = []
  fibers_names = []
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
      # E: bending modulus  
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
                        adaptive_num_points = 1,
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
  with open(read.output_name + '_fibers_info.txt', 'w') as f:
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
                                      verbose = flagVerbose, 
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
   
  # Files to save configurations before and after reparametrization
  buffering = max(1, min(fibers_types) * read.n_steps // read.n_save // 200)
  idBefore = open('before_reparam.txt','wb',buffering=buffering)
  idAfter = open('after_reparam.txt','wb',buffering=buffering)


  # Open config files
  if len(fibers_types) > 0:
    buffering = max(1, min(fibers_types) * read.n_steps // read.n_save // 200)
    f_fibers_ID = []
    for i, ID in enumerate(fibers_names):
      name = read.output_name + '_' + ID + 'fibers.txt'
      f = open(name, 'wb', buffering=buffering)
      f_fibers_ID.append(f)
    
  if len(body_types) > 0:   
    buffering = max(1, min(body_types) * read.n_steps // read.n_save // 200)
    f_bodies_ID = []
    for i, ID in enumerate(body_names):
      name = read.output_name + '_' + ID + 'clones.txt'
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

  # CHECKING IF LENGTH IS COMPUTED ACCURATELY
  length_computed = fibers[0].compute_length(fibers[0].x)
  length_given = fibers[0].length

  print('Computed length:', length_computed, ', Given length:', length_given)

  # TIME STEPPING
  start_time = time.time()  
  current_time = 0.0
  final_time = read.dt * read.n_steps
  time_save = 0.0
  step = 0
  steps_rejected = 0
  dt = read.dt 
  
  #ax = plt.axes(projection='3d')
  
  timer.timer('Entire_simulation')
  while current_time < final_time:

    # Save data if...
    if current_time >= time_save:
      time_save += read.n_save * read.dt
      elapsed_time = time.time() - start_time
      print('Integrator = ', read.scheme, ', step = ', step, ', time = ', current_time, ', dt = ', dt, ', wallclock time = ', time.time() - start_time)
      # For each type of structure save locations and orientations to one file
      # Save bodies
      bodies_offset = 0
      for i, ID in enumerate(body_names):
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
        for j in range(fibers_types[i]):
          np.savetxt(f_fibers_ID[i], np.ones((1,4), dtype=int)*fibers[fiber_offset + j].num_points)
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
    for fib_k, fib in enumerate(fibers):
      fib.xs = np.dot(fib.D_1, fib.x_new)
      fiber_error[fib_k] = abs(max(np.sqrt(fib.xs[1:-1,0]**2 + fib.xs[1:-1,1]**2 + fib.xs[1:-1,2]**2) - 1.0, key=abs))
      if fiber_error[fib_k] > fiber_error_max:
        fiber_error_max = fiber_error[fib_k]
    print('Error in Inextensibility = ', fiber_error_max)
    timer.timer('check_error')

    dt_old = dt
    max_acceptable_error = 1e-02

    timer.timer('update')
    # Update fibers
    for fib_k, fib in enumerate(fibers):
      #if fiber_error[fib_k] >= max_acceptable_error * 10:
      # Inextensibility Correction
      if False: 
        print('fiber = ', fib_k, ', error before correction = ', fiber_error[fib_k])
        fib.correct()
        print('fiber = ', fib_k, ', error after correction = ', abs(fib.compute_length(fib.x_new)-fib.length) / fib.length)
      else:
        fib.x = np.copy(fib.x_new)

      # Fiber reparametrization
      if True:
        np.savetxt(idBefore, np.ones((1,4), dtype=int)*fib.num_points)
        np.savetxt(idBefore, np.concatenate((fib.x, fib.tension[:,None]),axis=1))
        
        fib.reparametrize(30,6)
        
        np.savetxt(idAfter, np.ones((1,4), dtype=int)*fib.num_points)
        np.savetxt(idAfter, np.concatenate((fib.x, fib.tension[:,None]),axis=1))

    current_time += dt

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
    timer.timer('update')

    for fib in fibers:
      fib.beta = fib.beta_0 / dt
        
    step += 1     
    
    if False:
      for fib in fibers:
        ax.plot3D(fib.x[:,0],fib.x[:,1],fib.x[:,2])
        ax.set_xlim3d(-0.5,0.5)
        ax.set_ylim3d(-0.5,0.5)
        ax.set_zlim3d(0.,4.)
        #plt.pause(0.5)

  #plt.show()
  timer.timer('Entire_simulation')
 
  if current_time >= time_save:
    time_save += read.n_save * read.dt
    elapsed_time = time.time() - start_time
    print('Integrator = ', read.scheme, ', step = ', step, ', time = ', current_time, ', dt = ', dt, ', wallclock time = ', time.time() - start_time)
    # For each type of structure save locations and orientations to one file
    # Save bodies
    bodies_offset = 0
    for i, ID in enumerate(body_names):
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
      for j in range(fibers_types[i]):
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


  # Save wallclock time 
  with open(read.output_name + '.time', 'w') as f:
    f.write(str(time.time() - start_time) + '\n')
  timer.timer(' ', print_all = True, output_file=read.output_name+'.timers')
  print('\n\n\n# End')


