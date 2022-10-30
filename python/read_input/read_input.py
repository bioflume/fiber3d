'''
Simple class to read the input files to run a simulation.
'''
from __future__ import print_function
import numpy as np
import ntpath
import sys

class ReadInput(object):
  '''
  Simple class to read the input files to run a simulation.
  '''

  def __init__(self, entries):
    ''' Construnctor takes the name of the input file '''
    self.entries = entries
    self.input_file = entries
    self.options = {}
    number_of_structures = 0
    number_of_structures_fibers = 0
    number_of_structures_body_fibers = 0
    number_of_mm_surface = 0 

    # Read input file
    comment_symbols = ['#']   
    with open(self.input_file, 'r') as f:
      # Loop over lines
      for line in f:
        # Strip comments
        if comment_symbols[0] in line:
          line, comment = line.split(comment_symbols[0], 1)

        # Save options to dictionary, Value may be more than one word
        line = line.strip()
        if line != '':
          option, value = line.split(None, 1)
          if option == 'structure':
            option += str(number_of_structures)
            number_of_structures += 1
          elif option == 'structure_fibers':
            option += str(number_of_structures_fibers)
            number_of_structures_fibers += 1
          elif option == 'structure_body_fibers':
            option += str(number_of_structures_body_fibers)
            number_of_structures_body_fibers += 1
          elif option == 'mm_surface_structure':
            option += str(number_of_mm_surface)
            number_of_mm_surface += 1

          self.options[option] = value

    # Set option to file or default values
    self.n_steps = int(self.options.get('n_steps') or 0)
    self.initial_step = int(self.options.get('initial_step') or 0)
    self.n_save = int(self.options.get('n_save') or 1)
    self.dt = float(self.options.get('dt') or 0.0)
    self.eta = float(self.options.get('eta') or 1.0)
    self.kT = float(self.options.get('kT') or 1.0)
    self.scheme = str(self.options.get('scheme') or 'deterministic_forward_euler')
    self.output_name = str(self.options.get('output_name') or 'run')
    self.random_state = self.options.get('random_state')
    self.seed = self.options.get('seed')
    self.solver_tolerance = float(self.options.get('solver_tolerance') or 1e-08)
    self.save_clones = str(self.options.get('save_clones') or 'one_file_per_step')
    self.periodic_length = np.fromstring(self.options.get('periodic_length') or '0 0 0', sep=' ')
    self.num_points_fibers = int(self.options.get('num_points_fibers') or 0)
    self.E = float(self.options.get('E') or 0.0)
    self.length_fibers = float(self.options.get('length_fibers') or 0.0)
    self.epsilon = float(self.options.get('epsilon') or 1e-03)
    self.v_growth = float(self.options.get('v_growth') or 0.0)
    self.v_shrink = float(self.options.get('v_shrink') or 0.0)
    self.rate_catastrophe = float(self.options.get('rate_catastrophe') or 0.0)
    self.rate_rescue = float(self.options.get('rate_rescue') or 0.0)
    self.rate_seed = float(self.options.get('rate_seed') or 0.0)
    self.force_stall = float(self.options.get('force_stall') or 1.0)
    self.rate_catastrophe_stall = float(self.options.get('rate_catastrophe_stall') or 1.0)
    self.length_min = float(self.options.get('length_min') or 0.0)
    self.p_fmm = int(self.options.get('p_fmm') or 8)
    self.num_points_finite_diff = int(self.options.get('num_points_finite_diff') or 0) 
    self.mobility_vector_prod_implementation = str(self.options.get('mobility_vector_prod_implementation') or 'single_layer_double_layer')
    self.force_file = self.options.get('force_file')
    self.velocity_file = self.options.get('velocity_file')
    self.plot_velocity_field = np.fromstring(self.options.get('plot_velocity_field') or 'None', sep=' ')
    self.force_generator_config = self.options.get('force_generator_config') or None
    self.molecular_motor_config = self.options.get('molecular_motor_config') or None
    self.molecular_motor_radius = float(self.options.get('molecular_motor_radius') or 1.0)
    self.molecular_motor_speed = float(self.options.get('molecular_motor_speed') or 1.0)
    self.molecular_motor_force_stall = float(self.options.get('molecular_motor_force_stall') or 1.0)
    self.molecular_motor_spring_constant = float(self.options.get('molecular_motor_spring_constant') or 1.0)
    self.molecular_motor_rest_length = float(self.options.get('molecular_motor_rest_length') or 1.0)
    self.molecular_motor_bind_frequency = float(self.options.get('molecular_motor_bind_frequency') or 1.0)
    self.molecular_motor_unbind_frequency = float(self.options.get('molecular_motor_unbind_frequency') or 1.0)
    self.molecular_motor_kernel_sigma = float(self.options.get('molecular_motor_kernel_sigma') or 1.0)
    self.molecular_motor_attached_ends = self.options.get('molecular_motor_attached_ends') or None
    self.molecular_motor_head_config = self.options.get('molecular_motor_head_config') or None
    self.molecular_motor_base_config = self.options.get('molecular_motor_base_config') or None
    self.molecular_motor_s_head = self.options.get('molecular_motor_s_head') or None
    self.molecular_motor_s_base = self.options.get('molecular_motor_s_base') or None

    # Create list with [vertex_file, clones_file] for each structure
    self.structures = []
    for i in range(number_of_structures):
      option = 'structure' + str(i)
      structure_files = str.split(str(self.options.get(option)))
      self.structures.append(structure_files)

    # Create structures ID for each kind 
    self.structures_ID = []
    for struct in self.structures:
      # First, remove directory from structure name
      head, tail = ntpath.split(struct[1])
      # then, remove end (.clones)
      tail = tail[:-7]
      self.structures_ID.append(tail)

    # If we are restarting a simulation (initial_step > 0)
    # look for the .clones file in the output directory
    if self.initial_step > 0:
      for k, struct in enumerate(self.structures):
        recovery_file = self.output_name + '.'  + self.structures_ID[k] + '.' + str(self.initial_step).zfill(8) + '.clones'
        struct[1] = recovery_file


    # Create list with [fibers_file] for each structure_fiber
    self.structures_fibers = []
    for i in range(number_of_structures_fibers):
      option = 'structure_fibers' + str(i)
      structure_files = str(self.options.get(option))
      self.structures_fibers.append(structure_files)

    # Create structures ID for each kind (fibers)
    self.structures_fibers_ID = []
    for struct in self.structures_fibers:
      # First, remove directory from structure name
      head, tail = ntpath.split(struct)
      # then, remove end (.fibers)
      tail = tail[:-7]
      self.structures_fibers_ID.append(tail)

    # If we are restarting a simulation (initial_step > 0)
    # look for the .fibers file in the output directory
    if self.initial_step > 0:
      for k, struct in enumerate(self.structures_fibers):
        recovery_file = self.output_name + '.'  + self.structures_fibers_ID[k] + '.' + str(self.initial_step).zfill(8) + '.fibers'
        struct[1] = recovery_file


    # Create list with [files] for each structure_body_fibers
    self.structures_body_fibers = []
    for i in range(number_of_structures_body_fibers):
      option = 'structure_body_fibers' + str(i)
      structure_files = str.split(str(self.options.get(option)))
      self.structures_body_fibers.append(structure_files)

    # Create structures ID for each kind (body_fibers)
    self.structures_body_fibers_ID = []
    for struct in self.structures_body_fibers:
      # First, remove directory from structure name
      head, tail = ntpath.split(struct[1])
      # then, remove end (.clones)
      tail = tail[:-7]
      self.structures_body_fibers_ID.append(tail)

    # If we are restarting a simulation (initial_step > 0)
    # look for the .fibers file in the output directory
    if self.initial_step > 0:
      print('ERROR, code is not ready to restart from old configurations')
      sys.exit()
      for k, struct in enumerate(self.structures_body_fibers):
        recovery_file = self.output_name + '.'  + self.structures_body_fibers_ID[k] + '.' + str(self.initial_step).zfill(8) + '.fibers'
        struct[1] = recovery_file
    
    # MOVING MOLECULAR MOTOR STRUCTURES
    # Create list with [files] for each mm_surface_structure (includes mm_clones and mm_base mm_head attached_ends)
    self.mm_surface_structure = [] 
    for i in range(number_of_mm_surface):
      option = 'mm_surface_structure' + str(i)
      structure_files = str.split(str(self.options.get(option)))
      self.mm_surface_structure.append(structure_files)

    # Create structures ID for each kind (body_fibers)
    self.mm_surface_structure_ID = []
    for struct in self.mm_surface_structure:
      # First, remove directory from structure name
      head, tail = ntpath.split(struct[0])
      # this is the name of nucleus then (.clones)
      tail = tail[:-7]
      self.mm_surface_structure_ID.append(tail)

    return
    
