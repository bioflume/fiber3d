import numpy as np
import sys
sys.path.append('../')
from tstep import tstep_keav as tstep
from tstep import initialize_keav as initialize

if __name__ == '__main__':
  back_flow_dir =  'z' # parallel to the axis of symmetry
  # or perpendicular to the axis of symmetry
  #back_flow_dir = 'x'
  back_flow_u = 1
  body_shape = 'ellipsoid'
  # Prolate spheroid c > a = b
  body_a = 2*1.04
  body_b = 2*1.04
  body_c = 4*1.04

  filename = './fbgk/keavTest_par/run'
  #filename = './fbgk/keavTest_perp/run'
  # INPUT FILE: includes fiber, molecular motor parameters and files to read fiber, body, mm configs
  
  iComputeVelocity = False
  ncompute_vel = 500

  random_seed = 1
  time_step_scheme = 'time_step_hydro'
  Nbody = 500
  
  precompute_body_PC = True
  dt = 1E-4
  dt_max = 1E-4
  
  # SIMULATION'S OPTIONS AND PARAMETERS
  #  time_step_scheme = 'fibers_and_bodies_and_periphery_hydro_implicit_time_step',
  # Following options can be set without the input file
  options = initialize.set_options(time_step_scheme = time_step_scheme,
                                  order = 1,
                                  dt = dt,
                                  dt_min = dt,
                                  dt_max = dt_max,
                                  tol_tstep = 1e-1,
                                  tol_gmres = 1e-10,
                                  n_save = 1,
                                  output_name=filename,
                                  save_file = None,
                                  precompute_body_PC = precompute_body_PC,
                                  useFMM = False,
                                  Nblobs = Nbody,
                                  integration = 'trapz',
                                  iComputeVelocity = iComputeVelocity,
                                  ncompute_vel = ncompute_vel)

  prams = initialize.set_parameters(eta = 1,
      epsilon = 1e-03,
      final_time = 3*dt,
      body_shape = body_shape,
      body_a = body_a,
      body_b = body_b,
      body_c = body_c,
      back_flow_dir = back_flow_dir,
      back_flow_u = back_flow_u)

  # Create time stepping scheme
  tstep = tstep.tstep(prams,options,None,None,None,None)

  # Take time steps
  tstep.take_time_steps()


  print('\n\n\n# End')
