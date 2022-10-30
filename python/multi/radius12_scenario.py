import numpy as np
import sys
sys.path.append('../')
from tstep import tstep_scenario as tstep
from tstep import initialize_scenario as initialize
from read_input import read_input
from read_input import read_fibers_file
from read_input import read_vertex_file
from read_input import read_clones_file
from read_input import read_links_file


if __name__ == '__main__':
  radius = 12
  filename = './jaySims/scenario_runs2/radius' + str(radius) + '_scenario0/run'

  Ninit_hinged = 50 # initial number of hinged MTs
  how_many_kill = 40 # how many of them killed after a while
  when_kill = 4 # seconds after initialization
  how_many_create = 50 # how many new MTs created as hinged
  how_long_wait = 1.5 # seconds wait until the new ones created
  final_time = 15

  v_growth = 0.5
  scale_vg = 0.8

  input_file = './jaySims/input_files/mtoc_at_center.inputfile'
  time_step_scheme = 'time_step_hydro'
  peri_a = radius
  peri_b = radius
  peri_c = 25
  Nperiphery = 4000
  Nbody = 600
  body_quad_radius = 0.4
  iFixObjects = False
  periphery = 'ellipsoid'
  dt = 5E-3
  num_points_max = 256
  Nfiber = 16
  read = read_input.ReadInput(input_file)


  # SIMULATION'S OPTIONS AND PARAMETERS
  #  time_step_scheme = 'fibers_and_bodies_and_periphery_hydro_implicit_time_step',
  # Following options can be set without the input file
  options = initialize.set_options(adaptive_num_points = True,
                                  num_points = Nfiber,
                                  num_points_max = num_points_max,
                                  random_seed = 1,
                                  adaptive_time = False,
                                  time_step_scheme = time_step_scheme,
                                  order = 1,
                                  dt = dt,
                                  dt_min = dt,
                                  dt_max = dt,
                                  tol_tstep = 1e-1,
                                  tol_gmres = 1e-10,
                                  n_save = 25,
                                  output_name=filename,
                                  save_file = None,
                                  precompute_body_PC = True,
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
                                  repulsion = True)

  prams = initialize.set_parameters(eta = 1,
      v_growth = v_growth,
      scale_vg = scale_vg,
      epsilon = 1e-03,
      Efib = 20,
      final_time = final_time,
      fiber_body_attached = True,
      periphery = periphery,
      periphery_a = peri_a,
      periphery_b = peri_b,
      periphery_c = peri_c,
      minL = 0.5,
      max_nuc_sites = None,
      time_max = None,
      max_length = None,
      Ninit_hinged = Ninit_hinged,
      how_many_kill = how_many_kill,
      when_kill = when_kill,
      how_many_create = how_many_create,
      how_long_wait = how_long_wait)

 
  # Create time stepping scheme
  tstep = tstep.tstep(prams,options,input_file,None,None,None)

  # Take time steps
  tstep.take_time_steps()


  print('\n\n\n# End')
