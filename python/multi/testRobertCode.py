import numpy as np
import sys
sys.path.append('../')
from tstep import tstep_robert as tstep
from tstep import initialize_robert as initialize

if __name__ == '__main__':
  
  filename = './data/testRobertCode/run'
  cyto_force = -0.01
  iComputeVelocity = True
  ncompute_vel = 1000

  input_file = []
  random_seed = 1
  time_step_scheme = 'time_step_hydro'
  Nbody = 900
  body_quad_radius = 0.8
  mt_length = 10
  Nmts = 100
  dt = 1E-4
  num_points_max = 256
  Nfiber = 32
  bendingStiffness = 0.1
  body_radius = 1

  # SIMULATION'S OPTIONS AND PARAMETERS
  #  time_step_scheme = 'fibers_and_bodies_and_periphery_hydro_implicit_time_step',
  # Following options can be set without the input file
  options = initialize.set_options(adaptive_num_points = False,
                                  num_points = Nfiber,
                                  num_points_max = num_points_max,
                                  random_seed = random_seed,
                                  adaptive_time = False,
                                  time_step_scheme = time_step_scheme,
                                  dt = dt,
                                  dt_min = 1e-4,
                                  dt_max = 1e-1,
                                  tol_tstep = 1e-1,
                                  tol_gmres = 1e-10,
                                  n_save = 25,
                                  output_name=filename,
                                  save_file = None,
                                  precompute_body_PC = True,
                                  useFMM = True,
                                  Nblobs = Nbody,
                                  body_quadrature_radius = body_quad_radius,
                                  iupsample = False,
                                  uprate = 4,
                                  integration = 'trapz',
                                  iFixObjects = False,
                                  num_points_finite_diff = 4,
                                  inextensibility = 'penalty',
                                  iCytoPulling = True,
                                  dynInstability = False,
                                  iNoSliding = False,
                                  iExternalForce = False,
                                  min_body_ds = 0.01,
                                  fiber_ds = 0.5/32,
                                  repulsion = False,
                                  iComputeVelocity = iComputeVelocity,
                                  ncompute_vel = ncompute_vel)

  prams = initialize.set_parameters(eta = 1,
      epsilon = 1e-03,
      Efib = bendingStiffness,
      final_time = 2400,
      fiber_body_attached = True,
      growing = 1,
      minL = 0.5,
      mt_length = mt_length,
      Nmts = Nmts,
      body_radius = body_radius,
      cyto_force = cyto_force)

  # Create time stepping scheme
  tstep = tstep.tstep(prams,options,input_file,None,None,None)

  # Take time steps
  tstep.take_time_steps()


  print('\n\n\n# End')
