import numpy as np
import sys
sys.path.append('../')
from tstep import tstep_single as tstep
from tstep import initialize_single as initialize
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
  tstep0 = int(sys.argv[5])
  vscale = None
  filename = './jaySims/data_single_set' + str(int(set)) + '/radius' + str(int(radius)) + '_' + str(resume_step) + '_time' + str(tstep0) + '/run'
  velocity_file = './aveVel_s' + str(int(set)) + '_r' + str(int(radius)) + '_step' + str(resume_step) + '.mat'
  
  iComputeVelocity = False
  ncompute_vel = 1
  if set == 25:
    r0 = 1.5
  else:
    r0 = 0.5
  time_step_scheme = 'time_step_hydro'
  dt = 0.25
  num_points_max = 256
  Nfiber = 32

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
                                  dt_min = 1e-2,
                                  dt_max = 1e-1,
                                  tol_tstep = 1e-1,
                                  tol_gmres = 1e-10,
                                  n_save = 1,
                                  output_name = filename,
                                  save_file = None,
                                  useFMM = True,
                                  uprate = 4,
                                  integration = 'trapz',
                                  num_points_finite_diff = 4,
                                  inextensibility = 'penalty',
                                  iCytoPulling = False,
                                  dynInstability = True,
                                  iNoSliding = True,
                                  iExternalForce = False,
                                  fiber_ds = 0.5/32,
                                  iComputeVelocity = iComputeVelocity,
                                  ncompute_vel = ncompute_vel)

  prams = initialize.set_parameters(eta = 1,
      r0 = r0,
      velocity_file = velocity_file,
      vscale = vscale,
      scale_life_time = 0.2,
      scale_vg = 1,
      epsilon = 1e-03,
      Efib = 20,
      final_time = 2400,
      fiber_body_attached = False,
      confinement_radius = radius,
      tstep0 = tstep0,
      growing = 1,
      minL = 0.5,
      rate_catastrophe = 0.0625,
      v_growth = vgrowth,
      nucleation_rate = 32)

  # Create time stepping scheme
  tstep = tstep.tstep(prams,options,None,None,None,None)

  # Take time steps
  tstep.take_time_steps()


  print('\n\n\n# End')
