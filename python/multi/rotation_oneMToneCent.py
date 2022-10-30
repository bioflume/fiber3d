import numpy as np
import sys
sys.path.append('../')
from tstep import tstep_oneMToneCent as tstep
from tstep import initialize_oneMToneCent as initialize
from read_input import read_input
from read_input import read_fibers_file
from read_input import read_vertex_file
from read_input import read_clones_file
from read_input import read_links_file


if __name__ == '__main__':
  radius = float(sys.argv[1])
  vgrowth = float(sys.argv[2])
  scale = int(sys.argv[3])
  body_torque = scale * np.array([0, 0, 0.1])
  filename = './jaySims/singleMT/radius' + str(int(radius)) + '_scale' + str(scale) + '/run'
  # INPUT FILE: includes fiber, molecular motor parameters and files to read fiber, body, mm configs
  iComputeVelocity = True
  ncompute_vel = 100

  time_step_scheme = 'time_step_hydro'
  peri_a = radius
  peri_b = radius
  peri_c = 15
  Nperiphery = 4000
  Nbody = 400
  periphery = 'ellipsoid'
  precompute_body_PC = True
  dt = 2E-2
  dt_max = 5E-2
  time_max = None
  num_points_max = 256
  Nfiber = 16


  # SIMULATION'S OPTIONS AND PARAMETERS
  #  time_step_scheme = 'fibers_and_bodies_and_periphery_hydro_implicit_time_step',
  # Following options can be set without the input file
  options = initialize.set_options(adaptive_num_points = True,
                                  num_points = Nfiber,
                                  num_points_max = num_points_max,
                                  adaptive_time = False,
                                  time_step_scheme = time_step_scheme,
                                  order = 1,
                                  dt = dt,
                                  dt_min = 1e-4,
                                  tol_tstep = 1e-1,
                                  tol_gmres = 1e-10,
                                  n_save = 1,
                                  output_name=filename,
                                  save_file = None,
                                  precompute_body_PC = precompute_body_PC,
                                  useFMM = True,
                                  Nperiphery = Nperiphery,
                                  Nblobs = Nbody,
                                  iupsample = False,
                                  uprate = 4,
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
                                  iComputeVelocity = iComputeVelocity,
                                  ncompute_vel = ncompute_vel)

  prams = initialize.set_parameters(eta = 1,
      scale_life_time = 0.2,
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
      v_growth = vgrowth,
      body_torque = body_torque)

  # Create time stepping scheme
  tstep = tstep.tstep(prams,options,[],None,None,None)

  # Take time steps
  tstep.take_time_steps()


  print('\n\n\n# End')
