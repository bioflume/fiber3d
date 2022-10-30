import numpy as np
import sys
sys.path.append('../')
from tstep import tstep_new as tstep
from tstep import initialize_aFew as initialize
from read_input import read_input
from read_input import read_fibers_file
from read_input import read_vertex_file
from read_input import read_clones_file
from read_input import read_links_file


if __name__ == '__main__':

  # INPUT FILE: includes fiber, molecular motor parameters and files to read fiber, body, mm configs
  if True:
    random_seed = 1
    #input_file = './jaySims/input_files/mtoc_at_center.inputfile'
    input_file = './jaySims/input_files/radius13_2fibs_resume.inputfile'
    filename = './jaySims/data_Dec18/radius13_nfibers2_relax/run'
    nfibers = 2
    save_file = None
    repulsion = False
    time_step_scheme = 'time_step_hydro'
    useFMM = False
    peri_a = None
    peri_b = None
    peri_c = None
    Nperiphery = None
    Nbody = 400
    body_quad_radius = 0.4
    iFixObjects = False
    periphery = None
    precompute_body_PC = False
    max_length = None
    active_sites_idcs_file = None
    passive_sites_idcs_file = None
    fib_attached_to_body_file = None
    fib_nuc_site_idx_file = None
    fib_length_before_die_file = None
    dt = 1E-2
    dt_max = 1E-2
    time_max = None

  num_points_max = 128
  Nfiber = 32
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
                                  dt_min = 1e-3,
                                  dt_max = dt_max,
                                  tol_tstep = 1e-1,
                                  tol_gmres = 1e-10,
                                  n_save = 25,
                                  output_name=filename,
                                  save_file = save_file,
                                  precompute_body_PC = precompute_body_PC,
                                  useFMM = useFMM,
                                  Nperiphery = Nperiphery,
                                  Nblobs = Nbody,
                                  body_quadrature_radius = body_quad_radius,
                                  iupsample = False,
                                  integration = 'trapz',
                                  iFixObjects = iFixObjects,
                                  num_points_finite_diff = 4,
                                  inextensibility = 'penalty',
                                  iCytoPulling = False,
                                  dynInstability = False,
                                  iNoSliding = True,
                                  iExternalForce = False,
                                  min_body_ds = 0.01,
                                  fiber_ds = 0.5/32,
                                  repulsion = repulsion)

  prams = initialize.set_parameters(eta = 1,
      scale_life_time = 1,
      scale_vg = 0.75,
      epsilon = 1e-03,
      Efib = 20,
      final_time = 2400,
      fiber_body_attached = True,
      periphery = periphery,
      periphery_a = peri_a,
      periphery_b = peri_b,
      periphery_c = peri_c,
      growing = 1,
      rate_catastrophe = 1E-6,
      v_growth = 0.75,
      nucleation_rate = 0,
      max_nuc_sites = 1000,
      time_max = time_max,
      max_length = max_length,
      active_sites_idcs_file = active_sites_idcs_file,
      passive_sites_idcs_file = passive_sites_idcs_file,
      fib_attached_to_body_file = fib_attached_to_body_file,
      fib_nuc_site_idx_file = fib_nuc_site_idx_file,
      fib_length_before_die_file = fib_length_before_die_file)

  # Create time stepping scheme
  tstep = tstep.tstep(prams,options,input_file,None,None,None)

  # Take time steps
  tstep.take_time_steps()


  print('\n\n\n# End')
