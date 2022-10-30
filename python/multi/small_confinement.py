import numpy as np
import sys
sys.path.append('../')
from tstep import tstep as tstep
from tstep import initialize
from read_input import read_input
from read_input import read_fibers_file
from read_input import read_vertex_file
from read_input import read_clones_file
from read_input import read_links_file


if __name__ == '__main__':
  grow_mts = False
  # nuc rate is 15 (600 MTs), 7.5 (300MTs)
  # INPUT FILE: includes fiber, molecular motor parameters and files to read fiber, body, mm configs
  if grow_mts:
    input_file = './jaySims/input_files/mtoc_at_center.inputfile'
    filename = './jaySims/data/small_confinement_grow_cut/run'
    time_step_scheme = 'time_step_dry'
    repulsion = False
    useFMM = False
    peri_a = None
    peri_b = None
    peri_c = None
    Nperiphery = None
    Nbody = None
    body_quad_radius = None
    iFixObjects = True
    periphery = None
    precompute_body_PC = False
    max_length = 10
    active_sites_idcs_file = None
    passive_sites_idcs_file = None
    fib_attached_to_body_file = None
    fib_nuc_site_idx_file = None
    fib_length_before_die_file = None


  else:
    input_file = './jaySims/input_files/small_confinement_resume_cut.inputfile'
    filename = './jaySims/data/small_conf_cut_dry/run'
    #filename = './jaySims/data/small_conf_Lave20/run'
    #filename = './jaySims/data/small_conf_dry/run'
    repulsion = True
    time_step_scheme = 'time_step_dry'
    useFMM = True
    peri_a = 12.5
    peri_b = 12.5
    peri_c = 25
    Nperiphery = 4000
    Nbody = 400
    body_quad_radius = 0.4
    iFixObjects = False
    periphery = 'ellipsoid'
    max_length = None
    precompute_body_PC = True
    active_sites_idcs_file = './jaySims/Structures/small_conf_resume_cut/run-centrosome_at_center_active_site_idcs.clones'
    passive_sites_idcs_file = './jaySims/Structures/small_conf_resume_cut/run-centrosome_at_center_passive_site_idcs.clones'
    fib_attached_to_body_file = './jaySims/Structures/small_conf_resume_cut/run_centrosome_at_center_attached_to_body.fibers'
    fib_nuc_site_idx_file = './jaySims/Structures/small_conf_resume_cut/run_centrosome_at_center_nuc_site_idx.fibers'
    fib_length_before_die_file = './jaySims/Structures/small_conf_resume_cut/run_centrosome_at_center_length_before_die.fibers'

  Nfiber = 32
  dt = 1E-2
  read = read_input.ReadInput(input_file)


  # SIMULATION'S OPTIONS AND PARAMETERS
  #  time_step_scheme = 'fibers_and_bodies_and_periphery_hydro_implicit_time_step',
  # Following options can be set without the input file
  options = initialize.set_options(adaptive_num_points = True,
                                  num_points = Nfiber,
                                  random_seed = None,
                                  adaptive_time = True,
                                  time_step_scheme = time_step_scheme,
                                  order = 1,
                                  dt = dt,
                                  dt_min = 1e-3,
                                  dt_max = 5e-2,
                                  tol_tstep = 1e-1,
                                  tol_gmres = 1e-10,
                                  n_save = 25,
                                  output_name=filename,
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
                                  dynInstability = True,
                                  iNoSliding = True,
                                  iExternalForce = False,
                                  min_body_ds = 0.008,
                                  fiber_ds = 0.5/32,
                                  repulsion = repulsion)

  prams = initialize.set_parameters(eta = 1,
      epsilon = 1e-03,
      Efib = 20,
      final_time = 250000*dt,
      fiber_body_attached = True,
      periphery = periphery,
      periphery_a = peri_a,
      periphery_b = peri_b,
      periphery_c = peri_c,
      growing = 1,
      rate_catastrophe = 0.025,
      v_growth = 0.25,
      nucleation_rate = 7.5,
      max_nuc_sites = 750,
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
