import numpy as np
import sys
sys.path.append('../')
from tstep import time_step_container
from tstep import initialize_post as initialize
from tstep import postprocess_velocity


if __name__ == '__main__':
  set = int(sys.argv[1])
  radius = float(sys.argv[2])
  resume_step = int(sys.argv[3])
  radius_z = float(sys.argv[4])
  Nperiphery = int(sys.argv[5])
  step_start = int(sys.argv[6])
  step_end = int(sys.argv[7])
  nskip = int(sys.argv[8]) # multiplicative of 25
  compute_steps = np.arange(step_start, step_end+nskip, nskip)
  numRes = int(sys.argv[9])
  vgrowth = 0.5

  output_file = '/mnt/ceph/users/gkabacaoglu/JayAsterRotation/postprocess_velocity/set' + str(set) + '_radius' + str(int(radius)) + '_' + str(resume_step) + '/run'
  
  nucleating_site_file = './jaySims/Structures/centrosome.N1000.links'
  if set == 25 or set == 26 or set == 27:
    nucleating_site_file = './jaySims/Structures/3Ksites_R1p5.links'

  time_step_file, bodies_file, fibers_file = [], [], []

  for ires in np.arange(numRes):
    if ires == 0:
      filename = './jaySims/data_set' + str(set) + '/radius' + str(int(radius)) + '_' + str(resume_step) + '/run_centrosome_at_center_clones.txt'
      bodies_file.append(filename)
      filename = './jaySims/data_set' + str(set) + '/radius' + str(int(radius)) + '_' + str(resume_step) + '/run_centrosome_at_center_fibers.txt'
      fibers_file.append(filename)
      filename = './jaySims/data_set' + str(set) + '/radius' + str(int(radius)) + '_' + str(resume_step) + '/run_time_system_size.txt'
      time_step_file.append(filename)
    elif ires == 1:
      filename = './jaySims/data_set' + str(set) + '_resume/radius' + str(int(radius)) + '_' + str(resume_step) + '/run_centrosome_at_center_clones.txt'
      bodies_file.append(filename)
      filename = './jaySims/data_set' + str(set) + '_resume/radius' + str(int(radius)) + '_' + str(resume_step) + '/run_centrosome_at_center_fibers.txt'
      fibers_file.append(filename)
      filename = './jaySims/data_set' + str(set) + '_resume/radius' + str(int(radius)) + '_' + str(resume_step) + '/run_time_system_size.txt'
      time_step_file.append(filename)
    else:
      filename = './jaySims/data_set' + str(set) + '_resume' + str(ires) + '/radius' + str(int(radius)) + '_' + str(resume_step) + '/run_centrosome_at_center_clones.txt'
      bodies_file.append(filename)
      filename = './jaySims/data_set' + str(set) + '_resume' + str(ires) + '/radius' + str(int(radius)) + '_' + str(resume_step) + '/run_centrosome_at_center_fibers.txt'
      fibers_file.append(filename)
      filename = './jaySims/data_set' + str(set) + '_resume' + str(ires) + '/radius' + str(int(radius)) + '_' + str(resume_step) + '/run_time_system_size.txt'
      time_step_file.append(filename)

  peri_a = radius
  peri_b = radius
  peri_c = radius_z
  periphery = 'ellipsoid'

  Nbody = 800
  body_quad_radius = 0.4
  if set == 25 or set == 26 or set == 27 or set == 28:
    body_quad_radius = 1.25

  precompute_body_PC = True
  dt = 1E-2
  num_points_max = 256
  Nfiber = 64


  # SIMULATION'S OPTIONS AND PARAMETERS
  #  time_step_scheme = 'fibers_and_bodies_and_periphery_hydro_implicit_time_step',
  # Following options can be set without the input file
  options = initialize.set_options(num_points = Nfiber,
                                  num_points_max = num_points_max,
                                  dt = dt,
                                  tol_gmres = 1e-10,
                                  n_save = 25,
                                  output_name=output_file,
                                  precompute_body_PC = precompute_body_PC,
                                  useFMM = True,
                                  Nperiphery = Nperiphery,
                                  Nblobs = Nbody,
                                  body_quadrature_radius = body_quad_radius,
                                  iupsample = False,
                                  uprate = 8,
                                  num_points_finite_diff = 4,
                                  inextensibility = 'penalty',
                                  iCytoPulling = False,
                                  dynInstability = True,
                                  iNoSliding = True,
                                  fiber_ds = 0.5/32,
                                  repulsion = False)

  prams = initialize.set_parameters(eta = 1,
      nucleating_site_file = nucleating_site_file,
      bodies_file = bodies_file,
      fibers_file = fibers_file,
      time_step_file = time_step_file,
      body_viscosity_scale = 1,
      scale_vg = 1,
      epsilon = 1e-03,
      Efib = 20,
      periphery = periphery,
      periphery_a = peri_a,
      periphery_b = peri_b,
      periphery_c = peri_c,
      v_growth = vgrowth,
      compute_steps = compute_steps)

  # Initialize the files
  time_steps, time_all, nsteps_all = initialize.initialize_from_file(options, prams)

  # Create postprocessing object
  postprocess = postprocess_velocity.postprocess_velocity(prams, options, time_steps, nsteps_all, time_all)

  # Compute velocities
  postprocess.take_time_steps()
