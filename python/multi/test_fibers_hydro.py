import numpy as np
import sys
sys.path.append('../')
from tstep import tstep_one as tstep
from tstep import initialize_one as initialize
from read_input import read_input

if __name__ == '__main__':
  print('# Start')
  # Set parameters
  filename = 'data/cyto_relax/' + sys.argv[1] +'/run'

  Efib = float(sys.argv[2])
  fib_length = 16
  motor_sigma = float(sys.argv[3])

  radius = 18
  resume_step = None

  vgrowth = 0
  vgrowth_scale = 0
  body_viscosity_scale = 2
  # INPUT FILE: includes fiber, molecular motor parameters and files to read fiber, body, mm configs
  iComputeVelocity = False
  ncompute_vel = 100

  iRelaxationRun = False
  when_relax = None

  input_file = './jaySims/input_files/mtoc_at_center.inputfile'
  resume_from_step = None
  occupied_sites_idcs_file = None
  active_sites_idcs_file = None
  passive_sites_idcs_file = None
  fib_sites_lengths_file = None
  site_idcs_nucleating_file = None
  site_idcs_dying_file = None
  site_idcs_hinged_file = None

  random_seed = 1
  time_step_scheme = 'time_step_hydro'
  peri_a = radius
  peri_b = radius
  peri_c = 25
  Nperiphery = 6000
  Nbody = 300
  body_quad_radius = 0.4
  iFixObjects = False
  periphery = None #'ellipsoid'
  max_length = None
  precompute_body_PC = False
  dt = 1E-2
  dt_max = 1E-2
  time_max = None
  num_points_max = 256
  Nfiber = 16
  read = read_input.ReadInput(input_file)


  # SIMULATION'S OPTIONS AND PARAMETERS
  #  time_step_scheme = 'fibers_and_bodies_and_periphery_hydro_implicit_time_step',
  # Following options can be set without the input file
  options = initialize.set_options(adaptive_num_points = True,
                                  num_points = Nfiber,
                                  num_points_max = num_points_max,
                                  random_seed = random_seed,
                                  adaptive_time = False,
                                  time_step_scheme = time_step_scheme,
                                  order = 1,
                                  dt = dt,
                                  dt_min = 1e-2,
                                  dt_max = dt_max,
                                  tol_tstep = 1e-1,
                                  tol_gmres = 1e-10,
                                  n_save = 25,
                                  output_name=filename,
                                  save_file = None,
                                  precompute_body_PC = precompute_body_PC,
                                  useFMM = False,
                                  Nperiphery = Nperiphery,
                                  Nblobs = Nbody,
                                  body_quadrature_radius = body_quad_radius,
                                  iupsample = False,
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
                                  repulsion = True,
                                  iComputeVelocity = iComputeVelocity,
                                  ncompute_vel = ncompute_vel)

  prams = initialize.set_parameters(eta = 1,
      scale_life_time = 0.2,
      fib_length = fib_length,
      motor_sigma = motor_sigma,
      body_viscosity_scale = body_viscosity_scale,
      scale_vg = vgrowth_scale,
      epsilon = 1e-03,
      Efib = Efib,
      final_time = 500,
      fiber_body_attached = True,
      periphery = periphery,
      periphery_a = peri_a,
      periphery_b = peri_b,
      periphery_c = peri_c,
      growing = 1,
      minL = 0.5,
      rate_catastrophe = 0.0625,
      v_growth = vgrowth,
      nucleation_rate = 32,
      max_nuc_sites = None,
      time_max = time_max,
      max_length = max_length,
      resume_from_step = resume_from_step,
      active_sites_idcs_file = active_sites_idcs_file,
      passive_sites_idcs_file = passive_sites_idcs_file,
      occupied_sites_idcs_file = occupied_sites_idcs_file,
      fib_sites_lengths_file = fib_sites_lengths_file,
      site_idcs_nucleating_file = site_idcs_nucleating_file,
      site_idcs_dying_file = site_idcs_dying_file,
      site_idcs_hinged_file = site_idcs_hinged_file,
      iRelaxationRun = iRelaxationRun,
      when_relax = when_relax)

  # Create time stepping scheme
  tstep = tstep.tstep(prams,options,input_file,None,None,None)

  # Take time steps
  tstep.take_time_steps()
