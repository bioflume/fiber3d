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
  
  # INPUT FILE: includes fiber, molecular motor parameters and files to read fiber, body, mm configs
  input_file = './brinkman_test/input_files/Nf' + sys.argv[1] +'_L' + sys.argv[2] + '.inputfile'
  read = read_input.ReadInput(input_file)

  Nfiber = int(sys.argv[3])
  NfiniteDiff = 4
  Nbody = int(sys.argv[4])
  Nperiphery = int(sys.argv[5])
  body_quad_radius = 4.0
  dt = float(sys.argv[6])

  filename = './brinkman_test/output/Nf' + sys.argv[1] + '_Lf' + sys.argv[2] + '/run'
  # Following options can be set without the input file
  options = initialize.set_options(adaptive_num_points = False,
                                  num_points = Nfiber,
                                  ireparam = False,
                                  adaptive_time = False,
                                  time_step_scheme = 'time_step_hydro',
                                  order = 1,
                                  dt = dt,
                                  dt_min = 1e-3,
                                  dt_max = 2,
                                  tol_tstep = 1e-4,
                                  tol_gmres = 1e-10,
                                  inonlocal = False,
                                  output_txt_files = True,
                                  n_save = 100,
                                  output_name=filename,
                                  precompute_body_PC = True,
                                  useFMM = True,
                                  Nperiphery = Nperiphery,
                                  Nblobs = Nbody,
                                  body_quadrature_radius = body_quad_radius,
                                  iupsample = False,
                                  integration = 'trapz',
                                  iFixObjects = False,
                                  iExternalForce = True,
                                  irelease = False,
                                  iCytoPulling = False,
                                  slipVelOn = False,
                                  cytoPull_Elongation = False,
                                  num_points_finite_diff = NfiniteDiff,
                                  inextensibility = 'penalty')

  prams = initialize.set_parameters(eta = 1.0, 
      Efib = 10,
      epsilon = 1e-03, 
      final_time = 1000*dt, 
      fiber_body_attached = True,
      periphery = 'sphere',
      periphery_radius = 30,
      growing = 1)

  # Create time stepping scheme
  tstep = tstep.tstep(prams,options,input_file,None,None,None)

  # Take time steps
  tstep.take_time_steps()


  print('\n\n\n# End')    
