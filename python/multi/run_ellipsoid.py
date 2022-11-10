import numpy as np
import sys
sys.path.append('../')
from tstep import tstep
from tstep import initialize

if __name__ == '__main__':
  clamp_oscil_mag = np.pi/4
  clamp_oscil_k = 1
  clamp_oscil_speed = 0.10
  body_shape = 'ellipsoid'
  body_a = 2
  body_b = 2
  body_c = 4 # largest radius
  velMeasureRad = 2 * (2*body_c) # velocity on a shell of radius r = 2 * b, b: the largest diameter of the swimmer 
  velMeasureP = 16 # Chebyshev orderks
  num_fibers = 175
  fiber_position = None
  #fiber_position = []
  #fiber_position.append([2.0,0.0,0.0])
  #fiber_position.append([-2.0,0.0,0.0])
  #fiber_position.append([0.0,2.0,0.0])
  #fiber_position.append([0.0,-2.0,0.0])
  #fiber_position.append([0.0,0.0,4.0])
  #fiber_position.append([0.0,0.0,-4.0])
  fiber_length = 1
  
  # IF RESUMING, otherwise NONE
  save_folder = None # '/work2/03353/gokberk/frontera/flagellaRuns/test1/'
  resume_fiber_file = None #save_folder + 'run_fibers_resume.fibers'
  resume_body_file = None #save_folder + 'run_body_resume.clones'
  resume_time_file = None #save_folder + 'run_time_system_size.txt'
  body_link_location_file = None # save_folder + '_links_location.txt'
  
  #info = np.loadtxt(resume_time_file, dtype = np.float64)
  resume_from_time = 0 #info[-1,1]
  
  filename = '/work2/03353/gokberk/frontera/flagellaRuns/test2/run'
  # INPUT FILE: includes fiber, molecular motor parameters and files to read fiber, body, mm configs
  iComputeVelocity = True
  ncompute_vel = 450*5

  random_seed = 32
  time_step_scheme = 'time_step_hydro'
  Nbody = 800
  
  precompute_body_PC = True
  dt = 1E-3
  dt_max = 1E-3
  num_points_max = 256
  Nfiber = 48
  

  # SIMULATION'S OPTIONS AND PARAMETERS
  #  time_step_scheme = 'fibers_and_bodies_and_periphery_hydro_implicit_time_step',
  # Following options can be set without the input file
  options = initialize.set_options(adaptive_num_points = False,
                                  num_points = Nfiber,
                                  num_points_max = num_points_max,
                                  random_seed = random_seed,
                                  adaptive_time = False,
                                  time_step_scheme = time_step_scheme,
                                  order = 1,
                                  dt = dt,
                                  dt_min = dt,
                                  dt_max = dt_max,
                                  tol_tstep = 1e-1,
                                  tol_gmres = 1e-10,
                                  n_save = 250,
                                  output_name=filename,
                                  save_file = None,
                                  precompute_body_PC = precompute_body_PC,
                                  useFMM = False,
                                  Nblobs = Nbody,
                                  iupsample = False,
                                  uprate = 4,
                                  integration = 'trapz',
                                  num_points_finite_diff = 4,
                                  inextensibility = 'penalty',
                                  min_body_ds = 0.01,
                                  fiber_ds = 0.5/32,
                                  repulsion = False,
                                  iComputeVelocity = iComputeVelocity,
                                  ncompute_vel = ncompute_vel)

  prams = initialize.set_parameters(eta = 1,
      epsilon = 1e-03,
      Efib = 1,
      final_time = 250,
      fiber_body_attached = True,
      fiber_length = fiber_length,
      num_fibers = num_fibers,
      fiber_position = fiber_position,
      body_shape = body_shape,
      body_a = body_a,
      body_b = body_b,
      body_c = body_c,
      clamp_oscil_mag = clamp_oscil_mag,
      clamp_oscil_k = clamp_oscil_k,
      clamp_oscil_speed = clamp_oscil_speed,
      velMeasureRad = velMeasureRad,
      resume_from_time = resume_from_time,
      resume_fiber_file = resume_fiber_file,
      resume_body_file = resume_body_file,
      body_link_location_file = body_link_location_file)

  # Create time stepping scheme
  tstep = tstep.tstep(prams,options,None,None,None,None)

  # Take time steps
  tstep.take_time_steps()


  print('\n\n\n# End')
