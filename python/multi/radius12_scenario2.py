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
  radius = sys.argv[1]
  resTime = sys.argv[2]
  Nmin_hinged = int(sys.argv[3])
  wait_time = sys.argv[4]
  random_seed = sys.argv[5]
  filename = './jaySims/scenario_radius' + radius + '_runs/RT' + resTime + '_WT' + wait_time + '_Nmin' + str(Nmin_hinged) + '_rand' + random_seed + '/run'

  Nmax_hinged = 35 # max. num. of hinged MTs

  residency_time = float(resTime) # how long MT can sit on cortex
  when_kill = 20 #3.5*residency_time # seconds after initialization
  how_long_wait = float(wait_time) #1.5*residency_time # seconds wait until the new ones created
  final_time = 2*when_kill + how_long_wait

  v_growth = 0.65
  scale_vg = 1

  input_file = './jaySims/input_files/mtoc_at_center.inputfile'
  time_step_scheme = 'time_step_hydro'
  peri_a = float(radius)
  peri_b = float(radius)
  peri_c = 25.0
  Nperiphery = 4000
  Nbody = 600
  body_quad_radius = 0.4
  iFixObjects = False
  periphery = 'ellipsoid'
  dt = 8E-3 # 8E-3
  num_points_max = 256
  Nfiber = 16
  read = read_input.ReadInput(input_file)


  # SIMULATION'S OPTIONS AND PARAMETERS
  #  time_step_scheme = 'fibers_and_bodies_and_periphery_hydro_implicit_time_step',
  # Following options can be set without the input file
  options = initialize.set_options(adaptive_num_points = True,
                                  num_points = Nfiber,
                                  num_points_max = num_points_max,
                                  random_seed = int(random_seed),
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
      Nmax_hinged = Nmax_hinged,
      Nmin_hinged = Nmin_hinged,
      residency_time = residency_time,
      when_kill = when_kill,
      how_long_wait = how_long_wait)


  # Create time stepping scheme
  tstep = tstep.tstep(prams,options,input_file,None,None,None)

  # Take time steps
  tstep.take_time_steps()


  print('\n\n\n# End')
