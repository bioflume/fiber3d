import numpy as np
import sys
sys.path.append('../')
from tstep import tstep_resume as tstep
from tstep import initialize_resume as initialize
from read_input import read_input
from read_input import read_fibers_file
from read_input import read_vertex_file
from read_input import read_clones_file
from read_input import read_links_file


if __name__ == '__main__':
  radius = float(sys.argv[1])
  resume_step = int(sys.argv[2])
  set = int(sys.argv[3])
  vgrowth = float(sys.argv[4])
  vgrowth_scale = float(sys.argv[5])

  body_viscosity_scale = 1
  #filename = './jaySims/data_set' + str(set) + '_new_resume/radius' + str(int(radius)) + '_' + str(resume_step) + '/run'
  filename = '/mnt/ceph/users/gkabacaoglu/JayAsterRotation/data_set' + str(int(set)) + '_resume/radius' + str(int(radius)) + '_' + str(resume_step) + '/run'
  # INPUT FILE: includes fiber, molecular motor parameters and files to read fiber, body, mm configs
  iComputeVelocity = True
  ncompute_vel = 1000

  iRelaxationRun = False
  when_relax = None

  #input_file = './jaySims/input_files/mtoc_at_center.inputfile'
  input_file = './jaySims/input_files/largeMTOC_at_center.inputfile'
  save_folder = '/mnt/ceph/users/gkabacaoglu/JayAsterRotation/data_set' + str(int(set)) + '/radius' + str(int(radius)) + '_' + str(resume_step) + '/'
  #input_file = './jaySims/input_files/largeMTOC_at_center.inputfile'
  if resume_step is not None:
    info = np.loadtxt(save_folder + 'run_time_system_size.txt', dtype = np.float64)
    resume_from_step = resume_step + int(info[-1,2])
    occupied_sites_idcs_file = save_folder + 'run-centrosome_at_center_occupied_site_idcs.clones'
    active_sites_idcs_file = save_folder + 'run-centrosome_at_center_active_site_idcs.clones'
    passive_sites_idcs_file = save_folder + 'run-centrosome_at_center_passive_site_idcs.clones'
    fib_sites_lengths_file = save_folder +  'run_centrosome_at_center_nuc_site_idx.fibers'
    site_idcs_nucleating_file = '/mnt/ceph/users/gkabacaoglu/time_series_set' + str(set) + '/radius' + str(int(radius)) + '/site_info_nucleating.txt'
    site_idcs_dying_file = '/mnt/ceph/users/gkabacaoglu/time_series_set' + str(set) + '/radius' + str(int(radius)) + '/site_info_dying.txt'
    site_idcs_hinged_file = '/mnt/ceph/users/gkabacaoglu/time_series_set' + str(set) + '/radius' + str(int(radius)) + '/site_info_hinged.txt'
    body_file = save_folder + 'run_centrosome_at_center_resume.clones'
    fibers_file = save_folder + 'run_centrosome_at_center_resume.fibers'

  random_seed = 1
  time_step_scheme = 'time_step_hydro'
  peri_a = radius
  peri_b = radius
  if radius == 15:
    peri_c = 20
    Nperiphery = 5000
  elif radius == 20:
    peri_c = 25
    Nperiphery = 6000
  elif radius == 25:
    peri_c = 33
    Nperiphery = 8000
  elif radius == 30:
    peri_c = 40
    Nperiphery = 9000
  
  #peri_c = np.floor(20/15*radius)
  #if radius <= 20:
  #  Nperiphery = 6000
  #elif radius <= 25:
  #  Nperiphery = 8000
  #else:
  #  Nperiphery = 9000

  peri_c = 15
  Nperiphery = 8000

  Nbody = 1000 #800
  body_quad_radius = 1.25 #0.4
  iFixObjects = False
  periphery = 'ellipsoid'
  max_length = None
  precompute_body_PC = True
  dt = 1E-2
  dt_max = 1E-2
  time_max = None
  num_points_max = 256
  Nfiber = 64 #48
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
                                  uprate = 2,
                                  iComputeVelocity = iComputeVelocity,
                                  ncompute_vel = ncompute_vel)

  prams = initialize.set_parameters(eta = 1,
      scale_life_time = 0.2,
      body_viscosity_scale = body_viscosity_scale,
      scale_vg = vgrowth_scale,
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
      when_relax = when_relax,
      body_file = body_file,
      fibers_file = fibers_file)

  # Create time stepping scheme
  tstep = tstep.tstep(prams,options,input_file,None,None,None)

  # Take time steps
  tstep.take_time_steps()


  print('\n\n\n# End')
