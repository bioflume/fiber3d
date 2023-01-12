import numpy as np
import sys
sys.path.append('../')
from tstep import tstep_sphere as tstep
from tstep import initialize_sphere as initialize

if __name__ == '__main__':
  body_shape = 'ellipsoid'
  body_r = None # largest radius
  body_a = 2
  body_b = 4
  body_c = 2
  filename = '/work2/03353/gokberk/frontera/perpEllipsoid/run'
 # perpEllipsoid
  random_seed = 32
  time_step_scheme = 'time_step_hydro'
  Nbody = 200
  
  precompute_body_PC = True
  dt = 1E-2
  dt_max = 1E-2

  

  # SIMULATION'S OPTIONS AND PARAMETERS
  #  time_step_scheme = 'fibers_and_bodies_and_periphery_hydro_implicit_time_step',
  # Following options can be set without the input file
  options = initialize.set_options(adaptive_num_points = False,
                                  random_seed = random_seed,
                                  adaptive_time = False,
                                  time_step_scheme = time_step_scheme,
                                  order = 1,
                                  dt = dt,
                                  dt_min = dt,
                                  dt_max = dt_max,
                                  tol_tstep = 1e-1,
                                  tol_gmres = 1e-10,
                                  n_save = 10,
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
                                  repulsion = False)

  prams = initialize.set_parameters(eta = 1,
      epsilon = 1e-03,
      Efib = 1,
      final_time = 250,
      fiber_body_attached = False,
      body_shape = body_shape,
      body_r = body_r,
      body_a = body_a,
      body_b = body_b,
      body_c = body_c)

  # Create time stepping scheme
  tstep = tstep.tstep(prams,options,None,None,None,None)

  # Take time steps
  tstep.take_time_steps()


  print('\n\n\n# End')
