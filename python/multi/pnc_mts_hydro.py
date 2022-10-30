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
  #input_file = './input_files/july15_centrosome_at_center_grown.inputfile'
  input_file = './input_files/fibers_on_cortex.inputfile'
  #input_file = './input_files/july15_centrosome_at_halfway_grown.inputfile'
  #input_file = './input_files/single_fiber_hydro.inputfile'

  read = read_input.ReadInput(input_file)

  Nfiber = 48
  NfiniteDiff = 4
  Nbody = 1200 #1200, 4000
  Nperiphery = 8000 # 8000, 10000
  body_quad_radius = 0.4
  dt = 1E-2
  #filename = 'data/centInSphereCenterGrown/run'
  #filename = 'data/fibers_cortex_2K_newParams_pointB_belowBifur_Resume3/run'
  filename = 'data/fibers_cortex_2K_sigma45/run'
  #filename = 'data/fibers_cortex_2K_FIBO_pointB_belowBifur_Resume1/run'
  #filename = 'data/fibers_cortex_2K_newParams_pointB/run'
  #filename = 'data/PNC_MT_CORTEX/singleMT/run'
  
  # SIMULATION'S OPTIONS AND PARAMETERS
  # adaptive_num_points = flag for using adaptive fiber resolution (needed when polymerization is on)
  # ireparam = flag for fiber reparameterization (needed when resolution is low)
  # filtering = flag for de-aliasing (should be on always)
  # adaptive_time = flag for adaptive time stepping
  # time_step_scheme = scheme to take time steps (fibers? bodies? hydro?)
  # order = time stepping order (only 1 and 2 exist, 2 is less stable)
  # dt = time step size (if fixed, otherwise initial step size)
  # dt_min = minimum time step size allowed in adaptive stepping
  # dt_max = maximum time step size allowed in adaptive stepping
  # tol_tstep = tolerance to accept solution when taking time steps (for inextensibility error)

  # tol_gmres = tolerance for GMRES
  # inonlocal = flag for including nonlocal part of the self-mobility of fibers
  # igrowing = flag for fibers whether they are growing or not (probably not used)
  # random_seed = random seeding, if 1, then always the same series
  # output_txt_files = flag for saving data in txt files to use in MATLAB
  # output_name = name to save outputs


  #  time_step_scheme = 'fibers_and_bodies_and_periphery_hydro_implicit_time_step',
  # Following options can be set without the input file
  options = initialize.set_options(adaptive_num_points = False,
                                  num_points = Nfiber,
                                  ireparam = False,
                                  adaptive_time = False,
                                  time_step_scheme = 'time_step_hydro',
                                  order = 1,
                                  dt = dt,
                                  dt_min = 1e-4,
                                  dt_max = 1e-2,
                                  tol_tstep = 1e-2,
                                  tol_gmres = 1e-10,
                                  inonlocal = False,
                                  output_txt_files = True,
                                  n_save = 50,
                                  output_name=filename,
                                  precompute_body_PC = True,
                                  useFMM = True,
                                  Nperiphery = Nperiphery,
                                  Nblobs = Nbody,
                                  body_quadrature_radius = body_quad_radius,
                                  iupsample = False,
                                  integration = 'trapz',
                                  iFixObjects = False,
                                  iExternalForce = False,
                                  irelease = False,
                                  num_points_finite_diff = NfiniteDiff,
                                  inextensibility = 'penalty',
                                  iCytoPulling = True,
                                  iFiberOnly = True,
                                  belowBifur = True)

  prams = initialize.set_parameters(eta = 1.0, 
      epsilon = 1e-03,
      Efib = 2.5E-3,
      final_time = 250000*dt,
      fiber_body_attached = False,
      periphery = 'sphere',
      periphery_radius = 5.2,
      growing = 0)

  # Create time stepping scheme
  tstep = tstep.tstep(prams,options,input_file,None,None,None)

  # Take time steps
  tstep.take_time_steps()


  print('\n\n\n# End')    
