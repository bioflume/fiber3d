import numpy as np
import sys
sys.path.append('../')
from tstep import time_step_container
from tstep import initialize_post as initialize
from tstep import postprocess_velocity


if __name__ == '__main__':

  nskip = 900
  numRes = 2
  output_file = '/work2/03353/gokberk/frontera/flagellaRuns/postprocessVelocity/run'
  nucleating_site_file = '/work2/03353/gokberk/frontera/flagellaRuns/test5/run_links_location.txt'
  
  time_step_file, bodies_file, fibers_file = [], [], []
  for ires in np.arange(numRes):
    if ires == 0:
      filename = '/work2/03353/gokberk/frontera/flagellaRuns/test5/run_bodies.txt'
      bodies_file.append(filename)
      filename = '/work2/03353/gokberk/frontera/flagellaRuns/test5/run_fibers_fibers.txt'
      fibers_file.append(filename)
      filename = '/work2/03353/gokberk/frontera/flagellaRuns/test5/run_time_system_size.txt'
      time_step_file.append(filename)
    elif ires == 1:
      filename = '/work2/03353/gokberk/frontera/flagellaRuns/test5_resume/run_bodies.txt'
      bodies_file.append(filename)
      filename = '/work2/03353/gokberk/frontera/flagellaRuns/test5_resume/run_fibers_fibers.txt'
      fibers_file.append(filename)
      filename = '/work2/03353/gokberk/frontera/flagellaRuns/test5_resume/run_time_system_size.txt'
      time_step_file.append(filename)
    else:
      filename = '/work2/03353/gokberk/frontera/flagellaRuns/test5_resume2/run_bodies.txt'
      bodies_file.append(filename)
      filename = '/work2/03353/gokberk/frontera/flagellaRuns/test5_resume2/run_fibers_fibers.txt'
      fibers_file.append(filename)
      filename = '/work2/03353/gokberk/frontera/flagellaRuns/test5_resume2/run_time_system_size.txt'
      time_step_file.append(filename)

  Nbody = 800
  precompute_body_PC = True
  dt = 0.0025
  num_points_max = 256
  Nfiber = 64


  # SIMULATION'S OPTIONS AND PARAMETERS
  #  time_step_scheme = 'fibers_and_bodies_and_periphery_hydro_implicit_time_step',
  # Following options can be set without the input file
  options = initialize.set_options(adaptive_num_points = False,
                                  num_points = Nfiber,
                                  num_points_max = num_points_max,
                                  dt = dt,
                                  tol_gmres = 1e-10,
                                  n_save = 900,
                                  output_name=output_file,
                                  precompute_body_PC = precompute_body_PC,
                                  useFMM = False,
                                  Nblobs = Nbody,
                                  iupsample = False,
                                  uprate = 4,
                                  integration = 'trapz',
                                  num_points_finite_diff = 4,
                                  inextensibility = 'penalty',
                                  fiber_ds = 0.5/32)

  prams = initialize.set_parameters(eta = 1,
      Efib = 1,
      nucleating_site_file = nucleating_site_file,
      bodies_file = bodies_file,
      fibers_file = fibers_file,
      time_step_file = time_step_file,
      fiber_body_attached = True,
      body_a = 2,
      body_b = 2,
      body_c = 4,
      body_shape = 'ellipsoid',
      velMeasureP = 32,
      velMeasureRad = 2 * 2 * 4)

  # Initialize the files
  time_steps, time_all, nsteps_all = initialize.initialize_from_file(options, prams)

  # Create postprocessing object
  postprocess = postprocess_velocity.postprocess_velocity(prams, options, time_steps, nsteps_all, time_all, nskip)

  # Compute velocities
  postprocess.take_time_steps()
