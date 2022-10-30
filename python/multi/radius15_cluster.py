import numpy as np
import sys
sys.path.append('../')
from tstep import tstep_cluster as tstep
from tstep import initialize_cluster as initialize
from read_input import read_input
from read_input import read_fibers_file
from read_input import read_vertex_file
from read_input import read_clones_file
from read_input import read_links_file


if __name__ == '__main__':
  resume_step = int(sys.argv[1])
  radius = 15
  mtoc_radius = float(sys.argv[2]); # microns
  ncompute_vel = 500
  filename = './jaySims/data_set16_MTOC4_cluster/radius15_' + str(resume_step) + '/run'
  # INPUT FILE: includes fiber, molecular motor parameters and files to read fiber, body, mm configs

  input_file = './jaySims/input_files/mtoc_at_center.inputfile'
  if resume_step is not None:
    resume_from_step= resume_step
    occupied_sites_idcs_file = '/mnt/ceph/users/gkabacaoglu/time_series_set16/radius' + str(radius) +'/occupied_site_idcs.step' + str(resume_step)
    active_sites_idcs_file = '/mnt/ceph/users/gkabacaoglu/time_series_set16/radius' + str(radius) + '/active_nuc_site_idcs.step' + str(resume_step)
    passive_sites_idcs_file = '/mnt/ceph/users/gkabacaoglu/time_series_set16/radius' + str(radius) + '/passive_nuc_site_idcs.step' + str(resume_step)
    fib_sites_lengths_file = '/mnt/ceph/users/gkabacaoglu/time_series_set16/radius' + str(radius) + '/fiber_lengths.step' + str(resume_step)
    site_idcs_nucleating_file = '/mnt/ceph/users/gkabacaoglu/time_series_set16/radius' + str(radius) + '/site_info_nucleating.txt'
    site_idcs_dying_file = '/mnt/ceph/users/gkabacaoglu/time_series_set16/radius' + str(radius) + '/site_info_dying.txt'
    site_idcs_hinged_file = '/mnt/ceph/users/gkabacaoglu/time_series_set16/radius' + str(radius) + '/site_info_hinged.txt'
  else:
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
  Nperiphery = 4000
  Nbody = 600
  body_quad_radius = mtoc_radius - 1
  iFixObjects = False
  periphery = 'ellipsoid'
  max_length = None
  precompute_body_PC = True
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
                                  useFMM = True,
                                  Nperiphery = Nperiphery,
                                  Nblobs = Nbody,
                                  body_quadrature_radius = body_quad_radius,
                                  iupsample = False,
                                  integration = 'trapz',
                                  iFixObjects = False,
                                  num_points_finite_diff = 4,
                                  inextensibility = 'penalty',
                                  iCytoPulling = False,
                                  dynInstability = True,
                                  iNoSliding = True,
                                  iExternalForce = False,
                                  min_body_ds = 0.01,
                                  fiber_ds = 0.5/32,
                                  repulsion = True,
                                  iComputeVelocity = True,
                                  ncompute_vel = ncompute_vel)

  prams = initialize.set_parameters(eta = 1,
      mtoc_radius = mtoc_radius,
      scale_life_time = 0.2,
      scale_vg = 1,
      epsilon = 1e-03,
      Efib = 20,
      final_time = 2400,
      fiber_body_attached = True,
      periphery = periphery,
      periphery_a = peri_a,
      periphery_b = peri_b,
      periphery_c = peri_c,
      growing = 1,
      minL = 0.5,
      rate_catastrophe = 0.0625,
      v_growth = 0.4,
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
      site_idcs_hinged_file = site_idcs_hinged_file)

  # Create time stepping scheme
  tstep = tstep.tstep(prams,options,input_file,None,None,None)

  # Take time steps
  tstep.take_time_steps()


  print('\n\n\n# End')
